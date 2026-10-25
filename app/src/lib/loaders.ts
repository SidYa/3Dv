import * as THREE from 'three';
import { invoke } from '@tauri-apps/api/core';

// Each format's loader is fetched the first time a file of that format is
// opened. A viewer window only ever needs one or two of
// them, so the rest are neither downloaded nor parsed at startup.
const LOADER_MODULES = {
    gltf: () => Promise.all([
        import('three/examples/jsm/loaders/GLTFLoader.js'),
        import('three/examples/jsm/loaders/DRACOLoader.js'),
        import('three/examples/jsm/loaders/KTX2Loader.js'),
        import('three/examples/jsm/libs/meshopt_decoder.module.js')
    ]),
    obj: () => Promise.all([
        import('three/examples/jsm/loaders/OBJLoader.js'),
        import('three/examples/jsm/loaders/MTLLoader.js')
    ]),
    fbx: () => import('three/examples/jsm/loaders/FBXLoader.js'),
    stl: () => import('three/examples/jsm/loaders/STLLoader.js'),
    ply: () => import('three/examples/jsm/loaders/PLYLoader.js'),
    dae: () => import('three/examples/jsm/loaders/ColladaLoader.js'),
    tds: () => import('three/examples/jsm/loaders/TDSLoader.js')
};

const pathUtils = {
    extname: (p) => {
        const base = p.split(/[\\/]/).pop();
        const dot = base.lastIndexOf('.');
        return dot > 0 ? base.slice(dot) : '';
    },
    dirname: (p) => p.substring(0, Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))),
    basename: (p) => p.split(/[\\/]/).pop()
};

// Models are fetched over Tauri's asset:// protocol, which serves them
// from disk in the backend. Like the old threedv-model:// scheme this
// sidesteps MAX_PATH limits. Each path segment is encoded separately and
// joined with '/', so a glTF's relative buffer/texture references resolve
// against the model's URL just as they would over http - encoding the
// whole path as one segment would break them. The scheme differs per
// platform the same way convertFileSrc's does: WebView2 only routes
// http://asset.localhost, WebKitGTK and WKWebView only asset://localhost.
const ASSET_ORIGIN = /Windows|Android/.test(navigator.userAgent)
    ? 'http://asset.localhost'
    : 'asset://localhost';

export function toModelUrl(nativePath: string) {
    const fwd = nativePath.replace(/\\/g, '/');
    const encoded = fwd.split('/').map((seg) => encodeURIComponent(seg)).join('/');
    return `${ASSET_ORIGIN}/${encoded}`;
}

function hasVertexColors(geometry) {
    return !!(geometry.hasAttribute && geometry.hasAttribute('color'));
}

// Exporters routinely omit normals, or write them as all zeros - STL does
// it by convention, and OBJ/Collada files converted by older tools do it
// by accident. A mesh without usable normals gets no shading at all and
// renders as a flat white silhouette, so they are recomputed here.
function needsNormals(geometry) {
    if (!geometry.hasAttribute || !geometry.hasAttribute('position')) return false;
    if (!geometry.hasAttribute('normal')) return true;

    const normals = geometry.getAttribute('normal');
    const sampleCount = Math.min(normals.count, 64);
    for (let i = 0; i < sampleCount; i++) {
        if (normals.getX(i) || normals.getY(i) || normals.getZ(i)) return false;
    }
    return true;
}

function ensureNormals(object) {
    object.traverse((child) => {
        if (child.isMesh && child.geometry && needsNormals(child.geometry)) {
            child.geometry.computeVertexNormals();
        }
    });
}

function isTextureReady(tex) {
    if (!tex) return true;
    if (tex.isRenderTargetTexture || tex.isCubeTexture || tex.isCompressedTexture || tex.isDataTexture) return true;
    const img = tex.image;
    if (!img) return false;
    if (typeof img.complete === 'boolean') {
        return img.complete && (img.naturalWidth > 0 || img.width > 0);
    }
    if (typeof ImageBitmap !== 'undefined' && img instanceof ImageBitmap) return true;
    if (typeof OffscreenCanvas !== 'undefined' && img instanceof OffscreenCanvas) return true;
    if (img instanceof HTMLCanvasElement) return true;
    if (img instanceof HTMLVideoElement) return img.readyState >= 2;
    if (img.data && (img.width || img.height || img.data.length)) return true;
    return true;
}

export function areTexturesReady(model) {
    let ready = true;
    model.traverse((child) => {
        if (!ready || !child.isMesh) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const mat of materials) {
            if (!mat) continue;
            for (const key of Object.keys(mat)) {
                const value = mat[key];
                if (value && value.isTexture && !isTextureReady(value)) {
                    ready = false;
                    return;
                }
            }
        }
    });
    return ready;
}

let dracoLoader = null;
let ktx2Loader = null;

async function getGLTFLoader(glRenderer) {
    const [{ GLTFLoader }, { DRACOLoader }, { KTX2Loader }, { MeshoptDecoder }] = await LOADER_MODULES.gltf();
    if (!dracoLoader) {
        dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath(new URL('draco/', self.location.href).href);
    }
    if (!ktx2Loader) {
        ktx2Loader = new KTX2Loader();
        ktx2Loader.setTranscoderPath(new URL('basis/', self.location.href).href);
        if (glRenderer) ktx2Loader.detectSupport(glRenderer);
    }
    const loader = new GLTFLoader();
    loader.setDRACOLoader(dracoLoader);
    loader.setKTX2Loader(ktx2Loader);
    loader.setMeshoptDecoder(MeshoptDecoder);
    return loader;
}

async function loadOBJ(filePath, url, resolve, reject, onProgress) {
    const [{ OBJLoader }, { MTLLoader }] = await LOADER_MODULES.obj();
    const objOnly = () => new OBJLoader().load(url, resolve, onProgress, reject);
    const mtlPath = filePath.replace(/\.obj$/i, '.mtl');

    // HEAD, because this only asks whether the sidecar material file is
    // there; MTLLoader fetches the contents itself.
    fetch(toModelUrl(mtlPath), { method: 'HEAD' }).then(response => {
        if (!response.ok) { objOnly(); return; }

        const materialLoader = new MTLLoader();
        materialLoader.setPath(toModelUrl(pathUtils.dirname(mtlPath)) + '/');
        materialLoader.load(encodeURIComponent(pathUtils.basename(mtlPath)), (materials) => {
            materials.preload();
            const objLoader = new OBJLoader();
            objLoader.setMaterials(materials);
            objLoader.load(url, resolve, onProgress, reject);
        }, undefined, objOnly);
    }).catch(objOnly);
}

// --- Rust fast path -----------------------------------------------------------
// OBJ, PLY and STL are parsed by the Rust ports of these very loaders
// (core/render, command load_model): several times faster on large files and
// off the UI thread. The answer is one binary blob - "3DVG", the JSON length,
// the JSON describing meshes and materials, then the vertex data the typed
// arrays below view without copying. Textures load through TextureLoader
// exactly as the three.js loaders would.
const NATIVE_FORMATS = ['.obj', '.ply', '.stl'];
const NATIVE_MAGIC = 0x47564433; // "3DVG"

const isRemote = (url: string) => /^https?:/i.test(url);

// One three.js material from the JSON description (see viewer.rs).
function nativeMaterial(def: any, topology: string, dir: string, textures: Map<string, any>) {
    const common = {
        opacity: def.opacity,
        transparent: def.transparent,
        alphaTest: def.alphaTest,
        depthWrite: def.depthWrite,
        vertexColors: def.vertexColors,
        blending: def.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    };
    const color = new THREE.Color().setRGB(def.color[0], def.color[1], def.color[2]);
    const texture = (name: string, srgb: boolean) => {
        const t = def.maps[name];
        if (!t) return null;
        const key = t.source + (srgb ? ':srgb' : '') + ':' + t.matrix.join(',') + ':' + t.wrapS + t.wrapT + t.channel;
        let tex = textures.get(key);
        if (!tex) {
            // As MTLLoader / TDSLoader resolve them: the model folder plus the
            // reference as written.
            tex = textures.get('loader').load(isRemote(t.source) ? t.source : dir + t.source);
            if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
            tex.wrapS = t.wrapS;
            tex.wrapT = t.wrapT;
            tex.channel = t.channel;
            tex.matrixAutoUpdate = false;
            tex.matrix.set(...t.matrix);
            textures.set(key, tex);
        }
        return tex;
    };
    if (topology === 'points') {
        return new THREE.PointsMaterial({ ...common, color, size: def.size, sizeAttenuation: def.sizeAttenuation, map: texture('map', true) });
    }
    if (topology === 'lines') {
        return new THREE.LineBasicMaterial({ ...common, color });
    }
    const maps: Record<string, any> = {
        map: texture('map', true),
        emissiveMap: texture('emissiveMap', true),
        normalMap: texture('normalMap', false),
        bumpMap: texture('bumpMap', false),
        aoMap: texture('aoMap', false),
        alphaMap: texture('alphaMap', false),
        specularMap: texture('specularMap', false),
    };
    for (const k of Object.keys(maps)) if (!maps[k]) delete maps[k];
    const side = [THREE.FrontSide, THREE.BackSide, THREE.DoubleSide][def.side];
    const lit = {
        ...common,
        ...maps,
        color,
        side,
        flatShading: def.flatShading,
        emissive: new THREE.Color().setRGB(def.emissive[0], def.emissive[1], def.emissive[2]),
        bumpScale: def.bumpScale,
        normalScale: new THREE.Vector2(def.normalScale[0], def.normalScale[1]),
        aoMapIntensity: def.aoMapIntensity,
    };
    switch (def.type) {
        case 'standard': {
            const { specularMap, ...standard } = lit;
            return new THREE.MeshStandardMaterial({ ...standard, roughness: def.roughness, metalness: def.metalness });
        }
        case 'lambert':
            return new THREE.MeshLambertMaterial(lit);
        case 'basic': {
            const { emissiveMap, normalMap, bumpMap, ...basicMaps } = maps;
            return new THREE.MeshBasicMaterial({ ...common, ...basicMaps, color, side });
        }
        default:
            return new THREE.MeshPhongMaterial({
                ...lit,
                specular: new THREE.Color().setRGB(def.specular[0], def.specular[1], def.specular[2]),
                shininess: def.shininess,
            });
    }
}

// Builds the three.js objects the matching three loader would have built.
function buildNative(buffer: ArrayBuffer, filePath: string) {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== NATIVE_MAGIC) throw new Error('bad native model');
    const jsonLength = view.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, jsonLength)));
    const base = (8 + jsonLength + 3) & ~3;
    const floats = ([offset, length]: [number, number]) => new Float32Array(buffer, base + offset, length / 4);
    const uints = ([offset, length]: [number, number]) => new Uint32Array(buffer, base + offset, length / 4);
    const dir = toModelUrl(pathUtils.dirname(filePath)) + '/';
    const textures = new Map<string, any>([['loader', new THREE.TextureLoader()]]);
    const materials = new Map<string, any>();
    const material = (index: number, topology: string) => {
        const key = index + ':' + topology;
        if (!materials.has(key)) materials.set(key, nativeMaterial(header.materials[index], topology, dir, textures));
        return materials.get(key);
    };
    const group = new THREE.Group();
    for (const mesh of header.meshes) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(floats(mesh.position), 3));
        if (mesh.normal) geometry.setAttribute('normal', new THREE.BufferAttribute(floats(mesh.normal), 3));
        if (mesh.uv) geometry.setAttribute('uv', new THREE.BufferAttribute(floats(mesh.uv), 2));
        if (mesh.uv1) geometry.setAttribute('uv1', new THREE.BufferAttribute(floats(mesh.uv1), 2));
        if (mesh.color) geometry.setAttribute('color', new THREE.BufferAttribute(floats(mesh.color), 3));
        if (mesh.index) geometry.setIndex(new THREE.BufferAttribute(uints(mesh.index), 1));
        let meshMaterial;
        if (mesh.groups.length > 1) {
            meshMaterial = mesh.groups.map(([, , m]: number[]) => material(m, mesh.topology));
            mesh.groups.forEach(([start, count]: number[], i: number) => geometry.addGroup(start, count, i));
        } else {
            meshMaterial = material(mesh.groups[0][2], mesh.topology);
        }
        const Kind = mesh.topology === 'points' ? THREE.Points : mesh.topology === 'lines' ? THREE.LineSegments : THREE.Mesh;
        group.add(new Kind(geometry, meshMaterial));
    }
    return group;
}

// The model through Rust, or null to use the three.js loader (outside
// Tauri, or when the Rust side cannot read the file).
async function loadNative(filePath: string) {
    try {
        const buffer = await invoke<ArrayBuffer>('load_model', { path: filePath });
        return buildNative(buffer, filePath);
    } catch (e) {
        console.warn('Native loader unavailable, using three.js:', e);
        return null;
    }
}

// Loads any supported model file.
// Resolves with { object, animations }; rejects on failure.
// onProgress receives the underlying fetch's ProgressEvent, which the
// viewer turns into a percentage.
export function loadModelFile(filePath, glRenderer, onProgress) {
    return new Promise(async (resolve, reject) => {
        const ext = pathUtils.extname(filePath).toLowerCase();
        const url = toModelUrl(filePath);
        const asMesh = (g) => {
            const useVertexColors = hasVertexColors(g);
            return new THREE.Mesh(g, new THREE.MeshStandardMaterial({
                color: useVertexColors ? 0xffffff : 0x808080,
                vertexColors: useVertexColors,
                metalness: 0.05,
                roughness: 0.65
            }));
        };
        // A PLY without faces is a point cloud; drawing it as a mesh
        // produces an empty picture. The dot size is a placeholder - it is
        // measured in world units, so only the caller, which knows the scale
        // the model ends up at, can set it (see tunePointSize in scene.js).
        const asPoints = (g) => {
            const useVertexColors = hasVertexColors(g);
            return new THREE.Points(g, new THREE.PointsMaterial({
                size: 1,
                sizeAttenuation: true,
                color: useVertexColors ? 0xffffff : 0xcccccc,
                vertexColors: useVertexColors
            }));
        };
        const done = (object: any, animations?: any[]) => {
            ensureNormals(object);
            resolve({ object, animations: animations || object.animations || [] });
        };

        try {
            if (NATIVE_FORMATS.includes(ext)) {
                const object = await loadNative(filePath);
                if (object) {
                    done(object);
                    return;
                }
            }
            switch (ext) {
                case '.glb': case '.gltf':
                    (await getGLTFLoader(glRenderer)).load(url,
                        (gltf) => done(gltf.scene, gltf.animations), onProgress, reject);
                    break;
                case '.obj':
                    await loadOBJ(filePath, url, (obj) => done(obj), reject, onProgress);
                    break;
                case '.fbx': {
                    const { FBXLoader } = await LOADER_MODULES.fbx();
                    new FBXLoader().load(url, (obj) => done(obj), onProgress, reject);
                    break;
                }
                case '.stl': {
                    const { STLLoader } = await LOADER_MODULES.stl();
                    new STLLoader().load(url, (g) => done(asMesh(g)), onProgress, reject);
                    break;
                }
                case '.ply': {
                    const { PLYLoader } = await LOADER_MODULES.ply();
                    new PLYLoader().load(url,
                        (g) => done(g.index ? asMesh(g) : asPoints(g)), onProgress, reject);
                    break;
                }
                case '.dae': {
                    const { ColladaLoader } = await LOADER_MODULES.dae();
                    new ColladaLoader().load(url,
                        (collada) => done(collada.scene, collada.scene.animations), onProgress, reject);
                    break;
                }
                case '.3ds': {
                    const { TDSLoader } = await LOADER_MODULES.tds();
                    new TDSLoader().load(url, (obj) => done(obj), onProgress, reject);
                    break;
                }
                default:
                    reject(new Error('Unsupported format: ' + ext));
            }
        } catch (e) {
            reject(e);
        }
    });
}
