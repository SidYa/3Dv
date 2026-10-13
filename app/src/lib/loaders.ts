import * as THREE from 'three';

// Each format's loader is fetched the first time a file of that format is
// opened. A viewer or thumbnail process only ever needs one or two of
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
        const done = (object, animations) => {
            ensureNormals(object);
            resolve({ object, animations: animations || object.animations || [] });
        };

        try {
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
