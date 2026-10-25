// glTF 2.0 (.gltf and .glb): port of what three's GLTFLoader builds for a
// static thumbnail. Parity details:
//   - materials: metallic-roughness → MeshStandardMaterial (default material
//     metalness 1, roughness 1), KHR_materials_unlit → MeshBasicMaterial,
//     BLEND without depth writes, MASK cutoff 0.5 by default;
//   - a primitive without normals is drawn flat-shaded; without tangents
//     the normal map's green axis is flipped (derivative tangent frames);
//   - skinned meshes are posed by their joints' rest transforms (three's
//     "attached" bind mode makes the mesh node's own transform cancel out),
//     morph targets get their default weights;
//   - bounds come from the POSITION accessor's min/max widened by morph
//     displacements, as GLTFLoader's computeBounds; skinned meshes use the
//     posed vertices like SkinnedMesh.computeBoundingBox.
// Draco, meshopt, KTX2 and quantized files are left to the WebView renderer.

use std::path::Path;

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use gltf::mesh::Mode;
use gltf::texture::WrappingMode;

use super::{invalid, LoadError};
use crate::scene::{Aabb, AlphaMode, Material, Mesh, Placement, Scene, Side, TextureRef, Topology, UvTransform, Wrap};
use crate::texture::{Missing, Textures};

const UNSUPPORTED: &[&str] = &[
    "KHR_draco_mesh_compression",
    "EXT_meshopt_compression",
    "KHR_meshopt_compression",
    "KHR_texture_basisu",
    "KHR_mesh_quantization",
    "EXT_texture_avif",
];

pub fn load(path: &Path, textures: &mut Textures) -> Result<Scene, LoadError> {
    let bytes = std::fs::read(path)?;
    // The extensions are checked before validation, which rejects files
    // that require one the crate does not know.
    check_extensions(&bytes)?;
    let gltf = match gltf::Gltf::from_slice(&bytes) {
        Ok(g) => g,
        // GLTFLoader only warns about required extensions it does not know
        // (KHR_materials_pbrSpecularGlossiness, ...) and loads the rest; the
        // crate rejects the file. When that is the only complaint, parse it
        // again without validation.
        Err(gltf::Error::Validation(errors)) if errors.iter().all(|(_, e)| matches!(e, gltf::json::validation::Error::Unsupported)) => {
            gltf::Gltf::from_slice_without_validation(&bytes).map_err(|e| invalid(format!("glTF: {e}")))?
        }
        Err(e) => return Err(invalid(format!("glTF: {e}"))),
    };
    let document = gltf.document;
    let base = path.parent().unwrap_or(Path::new("."));
    let buffers = gltf::import_buffers(&document, Some(base), gltf.blob).map_err(|e| invalid(format!("glTF buffers: {e}")))?;

    let mut loader = Loader { scene: Scene::default(), textures, document: &document, buffers: &buffers, materials: Default::default(), images: Default::default() };
    let gscene = document.default_scene().or_else(|| document.scenes().next()).ok_or_else(|| invalid("glTF without a scene"))?;

    // World matrices of every node of the scene, for meshes and joints.
    let mut world = vec![None; document.nodes().count()];
    for node in gscene.nodes() {
        compute_world(&node, Mat4::IDENTITY, &mut world, 0);
    }
    for node in gscene.nodes() {
        loader.visit(&node, &world, 0)?;
    }
    Ok(loader.scene)
}

/// Reads extensionsRequired straight from the JSON (or the GLB's JSON chunk).
fn check_extensions(bytes: &[u8]) -> Result<(), LoadError> {
    let json: &[u8] = if bytes.starts_with(b"glTF") && bytes.len() >= 20 {
        let len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        bytes.get(20..20 + len).unwrap_or(&[])
    } else {
        bytes
    };
    let text = String::from_utf8_lossy(json);
    for ext in UNSUPPORTED {
        if text.contains(ext) {
            // Used only as an optional extension with an uncompressed
            // fallback is fine; required means the data needs it.
            if let Some(req) = text.find("\"extensionsRequired\"") {
                let tail = &text[req..];
                let end = tail.find(']').unwrap_or(tail.len());
                if tail[..end].contains(ext) {
                    return Err(LoadError::Unsupported((*ext).to_string()));
                }
            }
            if *ext == "KHR_mesh_quantization" {
                return Err(LoadError::Unsupported((*ext).to_string()));
            }
        }
    }
    Ok(())
}

fn local_matrix(node: &gltf::Node) -> Mat4 {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix),
        gltf::scene::Transform::Decomposed { translation, rotation, scale } => {
            Mat4::from_scale_rotation_translation(Vec3::from_array(scale), Quat::from_array(rotation), Vec3::from_array(translation))
        }
    }
}

const MAX_DEPTH: usize = 256;

fn compute_world(node: &gltf::Node, parent: Mat4, world: &mut Vec<Option<Mat4>>, depth: usize) {
    if depth > MAX_DEPTH {
        return;
    }
    let m = parent * local_matrix(node);
    if let Some(slot) = world.get_mut(node.index()) {
        *slot = Some(m);
    }
    for child in node.children() {
        compute_world(&child, m, world, depth + 1);
    }
}

struct Loader<'a> {
    scene: Scene,
    textures: &'a mut Textures,
    document: &'a gltf::Document,
    buffers: &'a [gltf::buffer::Data],
    /// (material index or None for the default, flat, derivative tangents,
    /// vertex colours, topology) → scene material.
    materials: std::collections::HashMap<(Option<usize>, bool, bool, bool, u8), usize>,
    images: std::collections::HashMap<usize, Option<usize>>,
}

impl<'a> Loader<'a> {
    fn visit(&mut self, node: &gltf::Node, world: &[Option<Mat4>], depth: usize) -> Result<(), LoadError> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        let m = world.get(node.index()).copied().flatten().unwrap_or(Mat4::IDENTITY);
        if let Some(mesh) = node.mesh() {
            let weights: Vec<f32> = node.weights().or_else(|| mesh.weights()).map(|w| w.to_vec()).unwrap_or_default();
            for primitive in mesh.primitives() {
                self.primitive(&primitive, node, m, world, &weights)?;
            }
        }
        for child in node.children() {
            self.visit(&child, world, depth + 1)?;
        }
        Ok(())
    }

    fn primitive(&mut self, primitive: &gltf::Primitive, node: &gltf::Node, node_world: Mat4, world: &[Option<Mat4>], weights: &[f32]) -> Result<(), LoadError> {
        let buffers = self.buffers;
        let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        let Some(positions) = reader.read_positions() else { return Ok(()) };
        let mut positions: Vec<[f32; 3]> = positions.collect();
        let n = positions.len();
        let mut normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|i| i.collect());
        let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|i| i.collect());
        let uv0: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|i| i.into_f32().collect());
        let uv1: Option<Vec<[f32; 2]>> = reader.read_tex_coords(1).map(|i| i.into_f32().collect());
        let colors: Option<Vec<[f32; 4]>> = reader.read_colors(0).map(|i| i.into_rgba_f32().collect());
        let indices: Option<Vec<u32>> = reader.read_indices().map(|i| i.into_u32().collect());
        if let Some(i) = &indices {
            if i.iter().any(|&v| v as usize >= n) {
                return Err(invalid("glTF index out of range"));
            }
        }
        let has_normals = normals.is_some();
        let has_tangents = tangents.is_some();
        let has_colors = colors.is_some();

        // Accessor bounds (GLTFLoader computeBounds), before morphing.
        let accessor = primitive.get(&gltf::Semantic::Positions);
        let mut local_box = accessor
            .as_ref()
            .and_then(|a| {
                let (min, max) = (a.min()?, a.max()?);
                let v = |j: &gltf::json::Value| -> Option<Vec3> {
                    let a = j.as_array()?;
                    Some(Vec3::new(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32))
                };
                Some(Aabb { min: v(&min)?, max: v(&max)? })
            })
            .unwrap_or_else(|| Aabb::from_points(positions.iter()));

        // Morph targets at their default weights. The box grows by the
        // largest displacement any single target allows, whatever the
        // weights (computeBounds' "reasonable size" compromise).
        let mut max_displacement = Vec3::ZERO;
        let target_bounds: Vec<Option<Vec3>> = primitive
            .morph_targets()
            .map(|t| {
                let a = t.positions()?;
                let (min, max) = (a.min()?, a.max()?);
                let (min, max) = (min.as_array()?.clone(), max.as_array()?.clone());
                let c = |k: usize| -> Option<f32> { Some(min.get(k)?.as_f64()?.abs().max(max.get(k)?.as_f64()?.abs()) as f32) };
                Some(Vec3::new(c(0)?, c(1)?, c(2)?))
            })
            .collect();
        for (i, (p, nrm, _)) in reader.read_morph_targets().enumerate() {
            let w = weights.get(i).copied().unwrap_or(0.0);
            if let Some(p) = p {
                let deltas: Vec<[f32; 3]> = p.collect();
                for (pos, d) in positions.iter_mut().zip(&deltas) {
                    *pos = (Vec3::from_array(*pos) + Vec3::from_array(*d) * w).to_array();
                }
                if let Some(Some(extent)) = target_bounds.get(i) {
                    max_displacement = max_displacement.max(*extent);
                }
            }
            if let (Some(nrm), Some(normals)) = (nrm, normals.as_mut()) {
                for (n, d) in normals.iter_mut().zip(nrm) {
                    *n = (Vec3::from_array(*n) + Vec3::from_array(d) * w).to_array();
                }
            }
        }
        local_box.min -= max_displacement;
        local_box.max += max_displacement;

        let topology = match primitive.mode() {
            Mode::Points => Topology::Points,
            Mode::Lines | Mode::LineStrip | Mode::LineLoop => Topology::Lines,
            Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan => Topology::Triangles,
        };
        let indices = expand_mode(primitive.mode(), indices, n);

        let mut mesh = Mesh::new(topology, positions);
        mesh.normals = normals;
        mesh.tangents = tangents;
        mesh.uv0 = uv0.filter(|v| v.len() == n);
        mesh.uv1 = uv1.filter(|v| v.len() == n);
        mesh.colors = colors.filter(|v| v.len() == n);
        mesh = match indices {
            Some(i) => mesh.with_indices(i),
            None => mesh,
        };
        let topo_key = match topology {
            Topology::Triangles => 0,
            Topology::Lines => 1,
            Topology::Points => 2,
        };
        let material = self.material(primitive.material().index(), !has_normals, !has_tangents, has_colors, topo_key);
        mesh = mesh.with_material(material);

        let placed = match node.skin().and_then(|s| self.skin(&s, &reader, &mesh, node_world, world)) {
            Some(skinned) => skinned,
            None => {
                let placement = Placement::new(node_world);
                let mut placed = placement.apply(mesh);
                placed.bounds = local_box.transformed(&node_world);
                placed
            }
        };
        self.scene.meshes.push(placed);
        Ok(())
    }

    /// Rest pose of a skinned primitive: Σ w·(joint world · inverse bind)·v,
    /// expressed back in the mesh node's space so the bounds come out as
    /// SkinnedMesh.computeBoundingBox's.
    fn skin<'s, F>(&self, skin: &gltf::Skin, reader: &gltf::mesh::Reader<'s, 's, F>, mesh: &Mesh, node_world: Mat4, world: &[Option<Mat4>]) -> Option<Mesh>
    where
        F: Clone + Fn(gltf::Buffer<'s>) -> Option<&'s [u8]>,
    {
        let joints: Vec<[u16; 4]> = reader.read_joints(0)?.into_u16().collect();
        let weights: Vec<[f32; 4]> = reader.read_weights(0)?.into_f32().collect();
        if joints.len() != mesh.positions.len() || weights.len() != mesh.positions.len() {
            return None;
        }
        let buffers = self.buffers;
        let ibms: Vec<Mat4> = skin
            .reader(|b| buffers.get(b.index()).map(|d| &d.0[..]))
            .read_inverse_bind_matrices()
            .map(|i| i.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_default();
        let bones: Vec<Mat4> = skin
            .joints()
            .enumerate()
            .map(|(i, j)| world.get(j.index()).copied().flatten().unwrap_or(Mat4::IDENTITY) * ibms.get(i).copied().unwrap_or(Mat4::IDENTITY))
            .collect();
        let to_local = if node_world.determinant().abs() > 1e-30 { node_world.inverse() } else { Mat4::IDENTITY };
        let mut out = mesh.clone();
        for (i, p) in out.positions.iter_mut().enumerate() {
            let s = skin_matrix(&bones, joints[i], weights[i]);
            *p = (to_local * s).transform_point3(Vec3::from_array(*p)).to_array();
        }
        if let Some(normals) = out.normals.as_mut() {
            for (i, n) in normals.iter_mut().enumerate() {
                let s = skin_matrix(&bones, joints[i], weights[i]);
                *n = (Mat3::from_mat4(to_local * s) * Vec3::from_array(*n)).to_array();
            }
        }
        if let Some(tangents) = out.tangents.as_mut() {
            for (i, t) in tangents.iter_mut().enumerate() {
                let s = skin_matrix(&bones, joints[i], weights[i]);
                let v = Mat3::from_mat4(to_local * s) * Vec3::new(t[0], t[1], t[2]);
                *t = [v.x, v.y, v.z, t[3]];
            }
        }
        Some(Placement::new(node_world).apply(out))
    }

    fn material(&mut self, index: Option<usize>, flat: bool, derivative_tangents: bool, vertex_colors: bool, topology: u8) -> usize {
        let key = (index, flat, derivative_tangents, vertex_colors, topology);
        if let Some(&m) = self.materials.get(&key) {
            return m;
        }
        let mut m = match index.and_then(|i| self.document.materials().nth(i)) {
            Some(def) => self.build_material(&def),
            // GLTFLoader's default material.
            None => Material { metalness: 1.0, roughness: 1.0, ..Material::standard() },
        };
        if flat {
            m.flat_shading = true;
        }
        if derivative_tangents {
            m.normal_scale.y = -m.normal_scale.y;
        }
        if vertex_colors {
            m.vertex_colors = true;
        }
        match topology {
            // PointsMaterial copy: colour, map, 1px dots.
            2 => {
                m = Material { color: m.color, opacity: m.opacity, map: m.map, vertex_colors: m.vertex_colors, alpha_mode: m.alpha_mode, depth_write: m.depth_write, ..Material::points(1.0, false) };
            }
            1 => {
                m = Material { color: m.color, opacity: m.opacity, map: m.map, vertex_colors: m.vertex_colors, alpha_mode: m.alpha_mode, depth_write: m.depth_write, side: m.side, ..Material::basic() };
            }
            _ => {}
        }
        let id = self.scene.add_material(m);
        self.materials.insert(key, id);
        id
    }

    fn build_material(&mut self, def: &gltf::Material) -> Material {
        let pbr = def.pbr_metallic_roughness();
        let unlit = def.unlit();
        let mut m = if unlit { Material::basic() } else { Material::standard() };
        let base = pbr.base_color_factor();
        m.color = Vec3::new(base[0], base[1], base[2]);
        m.opacity = base[3];
        m.map = pbr.base_color_texture().and_then(|t| self.texture_info(t.texture(), t.tex_coord(), t.texture_transform()));
        if !unlit {
            m.metalness = pbr.metallic_factor();
            m.roughness = pbr.roughness_factor();
            m.metal_rough_map = pbr.metallic_roughness_texture().and_then(|t| self.texture_info(t.texture(), t.tex_coord(), t.texture_transform()));
            if let Some(n) = def.normal_texture() {
                m.normal_map = self.texture_info(n.texture(), n.tex_coord(), None);
                m.normal_scale = Vec2::splat(n.scale());
            }
            if let Some(o) = def.occlusion_texture() {
                m.ao_map = self.texture_info(o.texture(), o.tex_coord(), None);
                m.ao_intensity = o.strength();
            }
            let e = def.emissive_factor();
            m.emissive = Vec3::from_array(e) * def.emissive_strength().unwrap_or(1.0);
            m.emissive_map = def.emissive_texture().and_then(|t| self.texture_info(t.texture(), t.tex_coord(), t.texture_transform()));
            // Transmission needs three's transmission pass; as an
            // approximation the surface lets the background through.
            if let Some(t) = def.transmission() {
                let f = t.transmission_factor();
                if f > 0.0 {
                    m.opacity *= (1.0 - f).max(0.1);
                    m.alpha_mode = AlphaMode::Blend;
                    m.depth_write = false;
                }
            }
        }
        if def.double_sided() {
            m.side = Side::Double;
        }
        match def.alpha_mode() {
            gltf::material::AlphaMode::Blend => {
                m.alpha_mode = AlphaMode::Blend;
                m.depth_write = false;
            }
            gltf::material::AlphaMode::Mask => m.alpha_mode = AlphaMode::Mask(def.alpha_cutoff().unwrap_or(0.5)),
            gltf::material::AlphaMode::Opaque => {
                if m.alpha_mode != AlphaMode::Blend {
                    m.alpha_mode = AlphaMode::Opaque;
                }
            }
        }
        m
    }

    fn texture_info(&mut self, texture: gltf::Texture, tex_coord: u32, transform: Option<gltf::texture::TextureTransform>) -> Option<TextureRef> {
        let image = self.image(texture.source().index())?;
        let mut t = TextureRef::new(image);
        t.uv_set = tex_coord.min(1) as u8;
        let sampler = texture.sampler();
        t.wrap_u = wrap(sampler.wrap_s());
        t.wrap_v = wrap(sampler.wrap_t());
        if let Some(tr) = transform {
            if let Some(tc) = tr.tex_coord() {
                t.uv_set = tc.min(1) as u8;
            }
            t.transform = UvTransform::three(Vec2::from_array(tr.offset()), Vec2::from_array(tr.scale()), tr.rotation(), Vec2::ZERO, false);
        }
        Some(t)
    }

    fn image(&mut self, index: usize) -> Option<usize> {
        if let Some(&cached) = self.images.get(&index) {
            return cached;
        }
        let image = self.document.images().nth(index)?;
        let key = format!("gltf-image-{index}");
        let result = match image.source() {
            gltf::image::Source::View { view, .. } => {
                let data = self.buffers.get(view.buffer().index())?;
                let bytes = data.0.get(view.offset()..view.offset() + view.length())?;
                self.textures.load_bytes(&mut self.scene, &key, bytes, Missing::Skip)
            }
            gltf::image::Source::Uri { uri, .. } => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let payload = rest.split_once(";base64,").map(|(_, b)| b)?;
                    let bytes = base64_decode(payload)?;
                    self.textures.load_bytes(&mut self.scene, &key, &bytes, Missing::Skip)
                } else {
                    self.textures.load_with(&mut self.scene, uri, false, Missing::Skip)
                }
            }
        };
        self.images.insert(index, result);
        result
    }
}

fn skin_matrix(bones: &[Mat4], joints: [u16; 4], weights: [f32; 4]) -> Mat4 {
    // normalizeSkinWeights(): scale to a sum of 1.
    let sum: f32 = weights.iter().sum();
    let scale = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    let mut acc = Mat4::ZERO;
    for k in 0..4 {
        let w = weights[k] * scale;
        if w != 0.0 {
            let b = bones.get(joints[k] as usize).copied().unwrap_or(Mat4::IDENTITY);
            acc += b * w;
        }
    }
    if sum > 0.0 {
        acc
    } else {
        Mat4::IDENTITY
    }
}

fn wrap(mode: WrappingMode) -> Wrap {
    match mode {
        WrappingMode::ClampToEdge => Wrap::Clamp,
        WrappingMode::MirroredRepeat => Wrap::Mirror,
        WrappingMode::Repeat => Wrap::Repeat,
    }
}

/// Strips, fans and line strips/loops as plain lists, the way three's
/// toTrianglesDrawMode and LineSegments see them.
fn expand_mode(mode: Mode, indices: Option<Vec<u32>>, n: usize) -> Option<Vec<u32>> {
    let seq = || -> Vec<u32> { (0..n as u32).collect() };
    match mode {
        Mode::TriangleStrip => {
            let src = indices.unwrap_or_else(seq);
            let mut out = Vec::with_capacity(src.len().saturating_sub(2) * 3);
            for i in 0..src.len().saturating_sub(2) {
                if i % 2 == 0 {
                    out.extend([src[i], src[i + 1], src[i + 2]]);
                } else {
                    out.extend([src[i + 2], src[i + 1], src[i]]);
                }
            }
            Some(out)
        }
        Mode::TriangleFan => {
            let src = indices.unwrap_or_else(seq);
            let mut out = Vec::new();
            for i in 1..src.len().saturating_sub(1) {
                out.extend([src[0], src[i], src[i + 1]]);
            }
            Some(out)
        }
        Mode::LineStrip | Mode::LineLoop => {
            let src = indices.unwrap_or_else(seq);
            let mut out = Vec::new();
            for w in src.windows(2) {
                out.extend([w[0], w[1]]);
            }
            if mode == Mode::LineLoop && src.len() > 2 {
                out.extend([src[src.len() - 1], src[0]]);
            }
            Some(out)
        }
        _ => indices,
    }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        } as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

