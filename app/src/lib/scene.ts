// Scene conventions of the viewer.
//
// The native thumbnail renderer (core/render, gpu.rs and shader.wgsl)
// reproduces this lighting, background and camera framing, so a cached
// thumbnail is the picture the viewer shows when the model is opened -
// change one, change the other.

import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

export const BACKGROUND_COLOR = 0x1a1a1a;

// Three-quarter view: a flat model seen head-on shows nothing.
const VIEW_DIRECTION = new THREE.Vector3(0.35, 0.3, 1).normalize();
const FRAME_PADDING = 1.08;

// Tone mapping is what keeps a plain white default material - what OBJ,
// Collada and FBX files without materials get - from clipping to a flat
// silhouette with no visible edges.
export function configureRenderer(renderer: THREE.WebGLRenderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1.0;
}

// One dominant key light gives the model readable form; the rest only
// keeps the shadowed sides from going black.
export function addStudioLights(scene: THREE.Scene) {
    const ambient = new THREE.AmbientLight(0xffffff, 0.12);
    scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1, 2, 3);
    scene.add(key);

    const back = new THREE.DirectionalLight(0xffffff, 0.25);
    back.position.set(-1, -0.5, -2);
    scene.add(back);

    const fill = new THREE.DirectionalLight(0xffffff, 0.3);
    fill.position.set(-1, 1, 1);
    scene.add(fill);

    return { ambient, key, back, fill };
}

// A neutral room reflection. Without it, metallic PBR materials - most
// of glTF - render as flat black.
export function applyEnvironment(scene: THREE.Scene, renderer: THREE.WebGLRenderer) {
    const pmrem = new THREE.PMREMGenerator(renderer);
    try {
        const target = pmrem.fromScene(new RoomEnvironment(), 0.04);
        scene.environment = target.texture;
        scene.environmentIntensity = 0.5;
        return target.texture;
    } catch (e) {
        console.warn('Environment map unavailable:', e);
        return null;
    } finally {
        pmrem.dispose();
    }
}

// Points the camera at the object and fits its bounding sphere, taking the
// narrower of the two field of view axes so nothing is cropped in a wide
// or tall window. Near/far follow the distance, which keeps depth
// precision usable for both millimetre- and kilometre-scale models.
// Returns the bounding sphere, or null when the object has no extent.
export function frameObject(camera: THREE.PerspectiveCamera, object: THREE.Object3D, target?: THREE.Vector3) {
    object.updateWorldMatrix(true, true);

    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return null;

    const sphere = box.getBoundingSphere(new THREE.Sphere());
    if (!isFinite(sphere.radius) || sphere.radius <= 0) return null;

    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * (camera.aspect || 1));
    const distance = (sphere.radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2)) * FRAME_PADDING;

    camera.position.copy(VIEW_DIRECTION).multiplyScalar(distance).add(sphere.center);
    camera.near = Math.max(distance / 1000, 1e-5);
    camera.far = (distance + sphere.radius) * 10;
    camera.lookAt(sphere.center);
    camera.updateProjectionMatrix();

    if (target) target.copy(sphere.center);

    return sphere;
}

// PointsMaterial sizes its dots in world units, and three does not apply the
// object's scale to them. Deriving the size from the units inside the file
// therefore breaks as soon as those are not metres: the same cloud authored
// in millimetres rendered as overlapping blobs, and in kilometres vanished.
// Taking it from the radius the model occupies on screen holds either way,
// and gives roughly one pixel per dot in a thumbnail and two or three in a
// window.
const POINT_SIZE_DIVISOR = 64;

export function tunePointSize(object: THREE.Object3D, worldRadius: number) {
    if (!(worldRadius > 0)) return;

    object.traverse((child: any) => {
        if (!child.isPoints || !child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) {
            if (material && material.isPointsMaterial) {
                material.size = worldRadius / POINT_SIZE_DIVISOR;
            }
        }
    });
}

function disposeMaterial(material: any) {
    for (const key of Object.keys(material)) {
        const value = material[key];
        if (value && value.isTexture) value.dispose();
    }
    material.dispose();
}

export function disposeObject(object: THREE.Object3D | null) {
    if (!object) return;
    object.traverse((child: any) => {
        if (child.geometry) child.geometry.dispose();
        if (!child.material) return;
        if (Array.isArray(child.material)) child.material.forEach(disposeMaterial);
        else disposeMaterial(child.material);
    });
}

// Meshes and points, with their materials flattened to a single array -
// used wherever a per-material tweak has to reach every material,
// including the array-material case.
export function eachMaterial(object: THREE.Object3D, callback: (material: any, child: THREE.Object3D) => void) {
    object.traverse((child: any) => {
        if (!child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) {
            if (material) callback(material, child);
        }
    });
}
