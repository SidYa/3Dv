// The flattened scene every loader produces and the renderer draws.
//
// It mirrors what three.js holds after one of its loaders ran: meshes, line
// segments and point clouds with their materials. The node hierarchy is
// already resolved - every position is in world space - because a thumbnail
// only ever draws the scene in its rest pose.
//
// Conventions shared by all loaders:
//   - colours are linear (three's working colour space); sRGB inputs are
//     converted with `srgb_to_linear` exactly where the matching three.js
//     loader calls ColorManagement / setRGB(..., SRGBColorSpace);
//   - texture coordinates address the image with v = 0 at its TOP row, like
//     glTF. Loaders whose three.js counterpart uploads textures with
//     flipY = true (everything but glTF) store 1 - v, see `flip_v`;
//   - a world matrix with a negative determinant reverses triangle winding,
//     which is what three's frontFaceCW does for mirrored objects.

use glam::{Mat3, Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Triangles,
    /// Independent segments, two vertices each (three's LineSegments).
    Lines,
    Points,
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) };

    pub fn is_empty(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y || self.max.z < self.min.z
    }

    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(&mut self, other: &Aabb) {
        if !other.is_empty() {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a [f32; 3]>) -> Aabb {
        let mut b = Aabb::EMPTY;
        for p in points {
            b.expand(Vec3::from_array(*p));
        }
        b
    }

    /// three's Box3.applyMatrix4: the box of the eight transformed corners.
    pub fn transformed(&self, m: &Mat4) -> Aabb {
        if self.is_empty() {
            return *self;
        }
        let mut b = Aabb::EMPTY;
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { self.min.x } else { self.max.x },
                if i & 2 == 0 { self.min.y } else { self.max.y },
                if i & 4 == 0 { self.min.z } else { self.max.z },
            );
            b.expand(m.transform_point3(c));
        }
        b
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Group {
    /// First index (indexed meshes) or first vertex (non-indexed ones).
    pub start: u32,
    pub count: u32,
    pub material: usize,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub topology: Topology,
    pub positions: Vec<[f32; 3]>,
    /// None, or one per position. Missing or all-zero normals of triangle
    /// meshes are recomputed by `Scene::finish` (loaders.ts ensureNormals).
    pub normals: Option<Vec<[f32; 3]>>,
    pub uv0: Option<Vec<[f32; 2]>>,
    pub uv1: Option<Vec<[f32; 2]>>,
    /// Linear RGBA; three's vertexColors only take effect when the material
    /// asks for them.
    pub colors: Option<Vec<[f32; 4]>>,
    /// xyz in world space, w = handedness (glTF TANGENT).
    pub tangents: Option<Vec<[f32; 4]>>,
    pub indices: Option<Vec<u32>>,
    /// Draw ranges. A mesh with a single material has one group spanning
    /// everything.
    pub groups: Vec<Group>,
    /// The box three's Box3.setFromObject() gets for this object: the
    /// geometry's local box transformed by the world matrix.
    pub bounds: Aabb,
    /// The object's world origin; three sorts transparent objects by it.
    pub origin: Vec3,
}

impl Mesh {
    pub fn vertex_count(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn element_count(&self) -> u32 {
        match &self.indices {
            Some(i) => i.len() as u32,
            None => self.positions.len() as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shading {
    /// MeshStandardMaterial / MeshPhysicalMaterial.
    Standard,
    /// MeshPhongMaterial.
    Phong,
    /// MeshLambertMaterial.
    Lambert,
    /// MeshBasicMaterial, LineBasicMaterial and PointsMaterial: no lighting.
    Basic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Front,
    Back,
    Double,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wrap {
    Repeat,
    Clamp,
    Mirror,
}

/// Row-major 2x3 affine map applied to a texture coordinate, three's
/// texture.matrix (Matrix3.setUvTransform) in image space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform(pub [f32; 6]);

impl UvTransform {
    pub const IDENTITY: UvTransform = UvTransform([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    /// three's Texture.updateMatrix(): offset, repeat, rotation and center.
    /// `flip_y` is true for textures three uploads with flipY (all but glTF):
    /// the transform is defined in three's v-up space, while our UVs are
    /// already flipped into image space, so it is conjugated with the flip.
    pub fn three(offset: Vec2, repeat: Vec2, rotation: f32, center: Vec2, flip_y: bool) -> UvTransform {
        let (s, c) = rotation.sin_cos();
        let (sx, sy) = (repeat.x, repeat.y);
        let (cx, cy) = (center.x, center.y);
        // Matrix3.setUvTransform
        let m = Mat3::from_cols(
            Vec3::new(sx * c, -sy * s, 0.0),
            Vec3::new(sx * s, sy * c, 0.0),
            Vec3::new(
                -sx * (c * cx + s * cy) + cx + offset.x,
                -sy * (-s * cx + c * cy) + cy + offset.y,
                1.0,
            ),
        );
        let m = if flip_y {
            let f = Mat3::from_cols(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 1.0));
            f * m * f
        } else {
            m
        };
        UvTransform([m.x_axis.x, m.y_axis.x, m.z_axis.x, m.x_axis.y, m.y_axis.y, m.z_axis.y])
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TextureRef {
    /// Index into `Scene::images`.
    pub image: usize,
    /// Which UV set the texture reads: 0 or 1.
    pub uv_set: u8,
    pub transform: UvTransform,
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
}

impl TextureRef {
    pub fn new(image: usize) -> TextureRef {
        TextureRef { image, uv_set: 0, transform: UvTransform::IDENTITY, wrap_u: Wrap::Repeat, wrap_v: Wrap::Repeat }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    /// alphaTest: fragments below the cutoff are discarded.
    Mask(f32),
    /// transparent = true: blended, drawn after the opaque objects.
    Blend,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub shading: Shading,
    /// Linear diffuse / base colour.
    pub color: Vec3,
    pub opacity: f32,
    /// Base colour texture (sRGB).
    pub map: Option<TextureRef>,
    pub vertex_colors: bool,
    /// Linear, emissiveIntensity already multiplied in.
    pub emissive: Vec3,
    /// sRGB.
    pub emissive_map: Option<TextureRef>,
    // Standard
    pub roughness: f32,
    pub metalness: f32,
    /// Linear; G = roughness, B = metalness (glTF metallicRoughnessTexture,
    /// assigned to both roughnessMap and metalnessMap in three).
    pub metal_rough_map: Option<TextureRef>,
    // Phong
    /// Linear.
    pub specular: Vec3,
    pub shininess: f32,
    /// R channel scales the specular highlight (three's specularMap).
    pub specular_map: Option<TextureRef>,
    // Common maps
    pub normal_map: Option<TextureRef>,
    pub normal_scale: Vec2,
    pub bump_map: Option<TextureRef>,
    pub bump_scale: f32,
    /// R channel.
    pub ao_map: Option<TextureRef>,
    pub ao_intensity: f32,
    /// G channel multiplies the alpha (three's alphaMap).
    pub alpha_map: Option<TextureRef>,
    pub alpha_mode: AlphaMode,
    pub depth_write: bool,
    pub side: Side,
    pub flat_shading: bool,
    /// PointsMaterial size and sizeAttenuation; `Scene::finish` sets the size
    /// the way tunePointSize() in scene.ts does.
    pub point_size: f32,
    pub size_attenuation: bool,
    /// three's material.blending = AdditiveBlending (3DS MAT_ADDITIVE).
    pub additive: bool,
}

impl Material {
    /// three's MeshStandardMaterial defaults.
    pub fn standard() -> Material {
        Material {
            shading: Shading::Standard,
            color: Vec3::ONE,
            opacity: 1.0,
            map: None,
            vertex_colors: false,
            emissive: Vec3::ZERO,
            emissive_map: None,
            roughness: 1.0,
            metalness: 0.0,
            metal_rough_map: None,
            specular: Vec3::splat(0x11 as f32 / 255.0),
            shininess: 30.0,
            specular_map: None,
            normal_map: None,
            normal_scale: Vec2::ONE,
            bump_map: None,
            bump_scale: 1.0,
            ao_map: None,
            ao_intensity: 1.0,
            alpha_map: None,
            alpha_mode: AlphaMode::Opaque,
            depth_write: true,
            side: Side::Front,
            flat_shading: false,
            point_size: 1.0,
            size_attenuation: true,
            additive: false,
        }
    }

    /// three's MeshPhongMaterial defaults: white, specular 0x111111,
    /// shininess 30.
    pub fn phong() -> Material {
        Material {
            shading: Shading::Phong,
            // 0x111111 is an sRGB hex colour in three.
            specular: Vec3::splat(srgb_to_linear(0x11 as f32 / 255.0)),
            ..Material::standard()
        }
    }

    pub fn lambert() -> Material {
        Material { shading: Shading::Lambert, ..Material::standard() }
    }

    pub fn basic() -> Material {
        Material { shading: Shading::Basic, ..Material::standard() }
    }

    /// three's PointsMaterial defaults.
    pub fn points(size: f32, size_attenuation: bool) -> Material {
        Material { shading: Shading::Basic, point_size: size, size_attenuation, ..Material::standard() }
    }

    /// Sets opacity and, below 1, the transparent flag - the pattern of most
    /// three loaders.
    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity;
        if opacity < 1.0 {
            self.alpha_mode = AlphaMode::Blend;
        }
    }

    pub fn is_transparent(&self) -> bool {
        self.alpha_mode == AlphaMode::Blend
    }
}

/// Decoded texture image, RGBA8, top row first. Already scaled down to what
/// a thumbnail can show (see texture.rs).
#[derive(Clone, Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Default, Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
}

/// three's Color.setRGB(r, g, b, SRGBColorSpace) for one channel.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c < 0.04045 {
        c * 0.0773993808
    } else {
        (c * 0.9478672986 + 0.0521327014).powf(2.4)
    }
}

pub fn srgb3(c: [f32; 3]) -> Vec3 {
    Vec3::new(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]))
}

/// From three's v-up texture space into image space (v = 0 at the top).
pub fn flip_v(uv: [f32; 2]) -> [f32; 2] {
    [uv[0], 1.0 - uv[1]]
}

/// Bakes a world matrix into geometry the way three would draw it. Positions
/// and the bounds are returned in world space; normals and tangents are
/// transformed with it; a mirroring matrix flips the winding of triangles.
pub struct Placement {
    pub world: Mat4,
    normal: Mat3,
    pub mirrored: bool,
}

impl Placement {
    pub fn new(world: Mat4) -> Placement {
        let m3 = Mat3::from_mat4(world);
        let det = m3.determinant();
        let normal = if det.abs() > 1e-30 { m3.inverse().transpose() } else { m3 };
        Placement { world, normal, mirrored: det < 0.0 }
    }

    pub fn identity() -> Placement {
        Placement::new(Mat4::IDENTITY)
    }

    pub fn position(&self, p: [f32; 3]) -> [f32; 3] {
        self.world.transform_point3(Vec3::from_array(p)).to_array()
    }

    pub fn normal(&self, n: [f32; 3]) -> [f32; 3] {
        let v = self.normal * Vec3::from_array(n);
        let len = v.length();
        if len > 0.0 {
            (v / len).to_array()
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    pub fn tangent(&self, t: [f32; 4]) -> [f32; 4] {
        let v = Mat3::from_mat4(self.world) * Vec3::new(t[0], t[1], t[2]);
        let v = v.normalize_or_zero();
        [v.x, v.y, v.z, t[3]]
    }

    /// Finishes a mesh whose attributes are still in the object's local
    /// space: computes the three-style bounds from the local positions,
    /// then moves everything into world space.
    pub fn apply(&self, mut mesh: Mesh) -> Mesh {
        let local = Aabb::from_points(mesh.positions.iter());
        mesh.bounds = local.transformed(&self.world);
        mesh.origin = self.world.transform_point3(Vec3::ZERO);
        for p in mesh.positions.iter_mut() {
            *p = self.position(*p);
        }
        if let Some(normals) = mesh.normals.as_mut() {
            for n in normals.iter_mut() {
                // Zero normals stay zero so ensure_normals still spots them.
                if n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0 {
                    *n = self.normal(*n);
                }
            }
        }
        if let Some(tangents) = mesh.tangents.as_mut() {
            for t in tangents.iter_mut() {
                *t = self.tangent(*t);
            }
        }
        if self.mirrored && mesh.topology == Topology::Triangles {
            reverse_winding(&mut mesh);
        }
        mesh
    }
}

impl Mesh {
    /// An empty mesh in local space, ready for `Placement::apply`.
    pub fn new(topology: Topology, positions: Vec<[f32; 3]>) -> Mesh {
        let count = positions.len() as u32;
        Mesh {
            topology,
            positions,
            normals: None,
            uv0: None,
            uv1: None,
            colors: None,
            tangents: None,
            indices: None,
            groups: vec![Group { start: 0, count, material: 0 }],
            bounds: Aabb::EMPTY,
            origin: Vec3::ZERO,
        }
    }

    pub fn with_indices(mut self, indices: Vec<u32>) -> Mesh {
        let count = indices.len() as u32;
        self.indices = Some(indices);
        self.groups = vec![Group { start: 0, count, material: self.groups.first().map_or(0, |g| g.material) }];
        self
    }

    pub fn with_material(mut self, material: usize) -> Mesh {
        for g in self.groups.iter_mut() {
            g.material = material;
        }
        self
    }
}

fn reverse_winding(mesh: &mut Mesh) {
    match mesh.indices.as_mut() {
        Some(indices) => {
            for tri in indices.chunks_exact_mut(3) {
                tri.swap(1, 2);
            }
        }
        None => {
            // Non-indexed: swap the attributes of vertices 1 and 2 of every
            // triangle.
            let n = mesh.positions.len() / 3 * 3;
            fn swap_all<T>(v: &mut [T], n: usize) {
                for i in (0..n.min(v.len() / 3 * 3)).step_by(3) {
                    v.swap(i + 1, i + 2);
                }
            }
            swap_all(&mut mesh.positions, n);
            if let Some(v) = mesh.normals.as_mut() {
                swap_all(v, n);
            }
            if let Some(v) = mesh.uv0.as_mut() {
                swap_all(v, n);
            }
            if let Some(v) = mesh.uv1.as_mut() {
                swap_all(v, n);
            }
            if let Some(v) = mesh.colors.as_mut() {
                swap_all(v, n);
            }
            if let Some(v) = mesh.tangents.as_mut() {
                swap_all(v, n);
            }
        }
    }
}

/// What the camera frames: three's Box3.setFromObject(model) and its
/// bounding sphere (scene.ts frameObject).
#[derive(Clone, Copy, Debug)]
pub struct Framing {
    pub bounds: Aabb,
    pub center: Vec3,
    pub radius: f32,
}

impl Scene {
    pub fn add_material(&mut self, material: Material) -> usize {
        self.materials.push(material);
        self.materials.len() - 1
    }

    pub fn add_image(&mut self, image: Image) -> usize {
        self.images.push(image);
        self.images.len() - 1
    }

    /// True when anything would draw pixels (scene.ts hasRenderableGeometry).
    pub fn has_renderable_geometry(&self) -> bool {
        self.meshes.iter().any(|m| !m.positions.is_empty() && m.groups.iter().any(|g| g.count > 0))
    }

    pub fn framing(&self) -> Option<Framing> {
        let mut bounds = Aabb::EMPTY;
        for m in &self.meshes {
            bounds.union(&m.bounds);
        }
        if bounds.is_empty() {
            return None;
        }
        let center = bounds.center();
        let radius = (bounds.max - bounds.min).length() * 0.5;
        if !radius.is_finite() || radius <= 0.0 {
            return None;
        }
        Some(Framing { bounds, center, radius })
    }

    /// The post-processing the viewer applies to every loaded model:
    /// ensureNormals() from loaders.ts, then tunePointSize() from scene.ts.
    /// Also drops groups that point at missing materials (three draws
    /// nothing for those).
    pub fn finish(&mut self) -> Option<Framing> {
        let material_count = self.materials.len();
        for mesh in self.meshes.iter_mut() {
            mesh.groups.retain(|g| g.material < material_count && g.count > 0);
            if mesh.topology == Topology::Triangles && needs_normals(mesh) {
                compute_vertex_normals(mesh);
            }
        }
        self.meshes.retain(|m| !m.groups.is_empty() && !m.positions.is_empty());
        let framing = self.framing()?;
        // tunePointSize: dots measured in world units, from the radius the
        // model occupies.
        let mut point_materials = vec![false; material_count];
        for mesh in &self.meshes {
            if mesh.topology == Topology::Points {
                for g in &mesh.groups {
                    point_materials[g.material] = true;
                }
            }
        }
        for (i, is_points) in point_materials.into_iter().enumerate() {
            if is_points {
                self.materials[i].point_size = framing.radius / 64.0;
            }
        }
        Some(framing)
    }
}

/// loaders.ts needsNormals(): missing, or the first 64 all zero.
fn needs_normals(mesh: &Mesh) -> bool {
    match &mesh.normals {
        None => true,
        Some(n) => n.len() != mesh.positions.len() || n.iter().take(64).all(|v| v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0),
    }
}

/// three's BufferGeometry.computeVertexNormals(): area-weighted face normals
/// summed per vertex for indexed geometry, flat face normals otherwise.
pub fn compute_vertex_normals(mesh: &mut Mesh) {
    let p = &mesh.positions;
    let mut normals = vec![Vec3::ZERO; p.len()];
    let face = |a: usize, b: usize, c: usize| {
        let (va, vb, vc) = (Vec3::from_array(p[a]), Vec3::from_array(p[b]), Vec3::from_array(p[c]));
        (vc - vb).cross(va - vb)
    };
    match &mesh.indices {
        Some(indices) => {
            for tri in indices.chunks_exact(3) {
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                if a >= p.len() || b >= p.len() || c >= p.len() {
                    continue;
                }
                let n = face(a, b, c);
                normals[a] += n;
                normals[b] += n;
                normals[c] += n;
            }
        }
        None => {
            for i in (0..p.len() / 3 * 3).step_by(3) {
                let n = face(i, i + 1, i + 2);
                normals[i] = n;
                normals[i + 1] = n;
                normals[i + 2] = n;
            }
        }
    }
    mesh.normals = Some(normals.into_iter().map(|n| n.normalize_or_zero().to_array()).collect());
}

/// Converts a 4-component colour with optional alpha into our layout.
pub fn rgba(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 1.0]
}
