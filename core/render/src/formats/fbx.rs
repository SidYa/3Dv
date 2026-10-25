// FBX, binary and ASCII, read with ufbx; materials and placement follow
// three's FBXLoader so the thumbnail matches the viewer:
//   - Phong or Lambert by ShadingModel, colours from the raw "Diffuse",
//     "DiffuseColor", "Specular", "Emissive"... properties (sRGB), opacity
//     from TransparencyFactor / Opacity the way the loader settles it;
//   - textures by the connection names the loader understands, embedded
//     content first, then the file next to the model;
//   - a Z-up file is rotated -90° about X and nothing else is converted
//     (no unit scaling, no handedness change);
//   - skinned meshes in the rest pose ufbx evaluates (bones at their local
//     transforms), bounds from the posed vertices as SkinnedMesh does.

use std::collections::HashMap;
use std::path::Path;

use glam::{Mat4, Vec2, Vec3};

use super::{invalid, LoadError};
use crate::scene::{flip_v, srgb3, srgb_to_linear, AlphaMode, Group, Material, Mesh, Placement, Scene, TextureRef, Topology, UvTransform, Wrap};
use crate::texture::{Missing, Textures};

pub fn load(path: &Path, textures: &mut Textures) -> Result<Scene, LoadError> {
    let path_str = path.to_str().ok_or_else(|| invalid("FBX path is not valid UTF-8"))?;
    let opts = ufbx::LoadOpts {
        evaluate_skinning: true,
        ignore_animation: true,
        load_external_files: false,
        // Missing normals stay missing: Scene::finish recomputes them the
        // way the viewer does.
        generate_missing_normals: false,
        ..Default::default()
    };
    let root = ufbx::load_file(path_str, opts).map_err(|e| invalid(format!("FBX: {}", e.description)))?;
    let fbx: &ufbx::Scene = &root;

    // three only corrects Z-up files, by rotating the whole scene.
    let up_z = matches!(fbx.settings.axes.up, ufbx::CoordinateAxis::PositiveZ | ufbx::CoordinateAxis::NegativeZ);
    let root_matrix = if up_z { Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) } else { Mat4::IDENTITY };

    let mut scene = Scene::default();
    let mut materials: HashMap<u32, usize> = HashMap::new();
    let mut default_material: Option<usize> = None;
    let mut fallback_white: Option<usize> = None;

    for node in fbx.nodes.iter() {
        let Some(mesh) = node.mesh.as_ref() else { continue };
        if mesh.num_faces == 0 {
            continue;
        }
        let node_to_world = root_matrix * mat4(&node.node_to_world);
        let geometry_to_node = mat4(&node.geometry_to_node);

        // Materials of this instance (FBX attaches them to the node).
        let instance_materials: Vec<&ufbx::Material> = if !node.materials.is_empty() { node.materials.iter().map(|m| &**m).collect() } else { mesh.materials.iter().map(|m| &**m).collect() };
        let has_colors = mesh.vertex_color.exists;
        let mut ids: Vec<usize> = Vec::with_capacity(instance_materials.len());
        for m in &instance_materials {
            let key = m.element.element_id;
            let id = match materials.get(&(key ^ if has_colors { 0x8000_0000 } else { 0 })) {
                Some(&id) => id,
                None => {
                    let mut mat = build_material(m, textures, &mut scene);
                    mat.vertex_colors = has_colors;
                    let id = scene.add_material(mat);
                    materials.insert(key ^ if has_colors { 0x8000_0000 } else { 0 }, id);
                    id
                }
            };
            ids.push(id);
        }
        if ids.is_empty() {
            // MeshPhongMaterial({ color: 0xcccccc }).
            let id = *default_material.get_or_insert_with(|| {
                let mut m = Material::phong();
                m.color = Vec3::splat(srgb_to_linear(0xcc as f32 / 255.0));
                m.vertex_colors = has_colors;
                scene.add_material(m)
            });
            ids.push(id);
        }

        let skinned = mesh.skinned_position.exists && !mesh.skin_deformers.is_empty();
        let built = build_mesh(mesh, &ids, &mut fallback_white, &mut scene, skinned);
        let Some(mut built) = built else { continue };

        let placed = if skinned {
            // ufbx's skinned positions are in world space unless
            // skinned_is_local; bring them into the node's space so the
            // bounds come out as SkinnedMesh.computeBoundingBox's.
            let to_node = if node_to_world.determinant().abs() > 1e-30 { node_to_world.inverse() } else { Mat4::IDENTITY };
            let skin_space = if mesh.skinned_is_local { geometry_to_node } else { to_node * root_matrix };
            transform_local(&mut built, &skin_space);
            Placement::new(node_to_world).apply(built)
        } else {
            transform_local(&mut built, &geometry_to_node);
            Placement::new(node_to_world).apply(built)
        };
        scene.meshes.push(placed);
    }
    Ok(scene)
}

fn mat4(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols_array(&[
        m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0,
        m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0,
        m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0,
        m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0,
    ])
}

/// Applies a matrix to positions and normals in place (three bakes the
/// geometric transform into the geometry).
fn transform_local(mesh: &mut Mesh, m: &Mat4) {
    if *m == Mat4::IDENTITY {
        return;
    }
    let p = Placement::new(*m);
    for v in mesh.positions.iter_mut() {
        *v = p.position(*v);
    }
    if let Some(normals) = mesh.normals.as_mut() {
        for n in normals.iter_mut() {
            if n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0 {
                *n = p.normal(*n);
            }
        }
    }
}

/// Unshared triangles grouped by material, like FBXLoader's geometry.
fn build_mesh(mesh: &ufbx::Mesh, ids: &[usize], fallback_white: &mut Option<usize>, scene: &mut Scene, skinned: bool) -> Option<Mesh> {
    let positions_attr = if skinned { &mesh.skinned_position } else { &mesh.vertex_position };
    let normals_attr = if skinned && mesh.skinned_normal.exists { &mesh.skinned_normal } else { &mesh.vertex_normal };
    let uv_attr = &mesh.vertex_uv;
    let uv1_attr = mesh.uv_sets.get(1).map(|s| &s.vertex_uv);
    let color_attr = &mesh.vertex_color;

    // Faces sorted by material, preserving file order within a material.
    let mut by_material: Vec<Vec<usize>> = vec![Vec::new(); ids.len() + 1];
    for (fi, _) in mesh.faces.iter().enumerate() {
        let m = mesh.face_material.get(fi).copied().unwrap_or(0) as usize;
        // A face pointing past the material list gets an extra default
        // MeshPhongMaterial, as FBXLoader's group sanitising does.
        let slot = if m < ids.len() { m } else { ids.len() };
        by_material[slot].push(fi);
    }

    let mut positions = Vec::with_capacity(mesh.num_triangles * 3);
    let mut normals = normals_attr.exists.then(|| Vec::with_capacity(mesh.num_triangles * 3));
    let mut uv0 = uv_attr.exists.then(|| Vec::with_capacity(mesh.num_triangles * 3));
    let mut uv1 = uv1_attr.filter(|a| a.exists).map(|_| Vec::with_capacity(mesh.num_triangles * 3));
    let mut colors = color_attr.exists.then(|| Vec::with_capacity(mesh.num_triangles * 3));
    let mut groups = Vec::new();
    let mut tri = vec![0u32; mesh.max_face_triangles * 3];

    for (slot, faces) in by_material.iter().enumerate() {
        if faces.is_empty() {
            continue;
        }
        let material = if slot < ids.len() {
            ids[slot]
        } else {
            *fallback_white.get_or_insert_with(|| scene.add_material(Material::phong()))
        };
        let start = positions.len() as u32;
        for &fi in faces {
            let face = mesh.faces[fi];
            if face.num_indices < 3 {
                continue;
            }
            let n = ufbx::triangulate_face(&mut tri, mesh, face) as usize;
            for &ix in &tri[..n * 3] {
                let ix = ix as usize;
                positions.push(vec3(attr3(positions_attr, ix)));
                if let Some(v) = normals.as_mut() {
                    v.push(vec3(attr3(normals_attr, ix)));
                }
                if let Some(v) = uv0.as_mut() {
                    v.push(flip_v(attr2(uv_attr, ix)));
                }
                if let (Some(v), Some(a)) = (uv1.as_mut(), uv1_attr) {
                    v.push(flip_v(attr2(a, ix)));
                }
                if let Some(v) = colors.as_mut() {
                    let c = attr4(color_attr, ix);
                    let lin = srgb3([c[0], c[1], c[2]]);
                    v.push([lin.x, lin.y, lin.z, c[3]]);
                }
            }
        }
        let count = positions.len() as u32 - start;
        if count > 0 {
            groups.push(Group { start, count, material });
        }
    }
    if positions.is_empty() {
        return None;
    }
    let mut out = Mesh::new(Topology::Triangles, positions);
    out.normals = normals;
    out.uv0 = uv0;
    out.uv1 = uv1;
    out.colors = colors;
    out.groups = groups;
    Some(out)
}

fn vec3(v: [f32; 3]) -> [f32; 3] {
    v
}

fn attr3(a: &ufbx::VertexVec3, ix: usize) -> [f32; 3] {
    let i = a.indices.get(ix).copied().unwrap_or(0) as usize;
    a.values.get(i).map(|v| [v.x as f32, v.y as f32, v.z as f32]).unwrap_or([0.0; 3])
}

fn attr2(a: &ufbx::VertexVec2, ix: usize) -> [f32; 2] {
    let i = a.indices.get(ix).copied().unwrap_or(0) as usize;
    a.values.get(i).map(|v| [v.x as f32, v.y as f32]).unwrap_or([0.0; 2])
}

fn attr4(a: &ufbx::VertexVec4, ix: usize) -> [f32; 4] {
    let i = a.indices.get(ix).copied().unwrap_or(0) as usize;
    a.values.get(i).map(|v| [v.x as f32, v.y as f32, v.z as f32, v.w as f32]).unwrap_or([1.0; 4])
}

/// An explicitly set property of the material node (FBXLoader does not see
/// template defaults).
fn prop<'a>(m: &'a ufbx::Material, name: &str) -> Option<&'a ufbx::Prop> {
    m.element.props.props.iter().find(|p| &*p.name == name)
}

fn prop_color(m: &ufbx::Material, name: &str) -> Option<Vec3> {
    let p = prop(m, name)?;
    Some(srgb3([p.value_vec4.x as f32, p.value_vec4.y as f32, p.value_vec4.z as f32]))
}

fn prop_real(m: &ufbx::Material, name: &str) -> Option<f32> {
    prop(m, name).map(|p| p.value_vec4.x as f32)
}

/// Is the property a colour (FBX type "Color" / "ColorRGB")? ufbx keeps the
/// original type name only in the DOM; a three-component value is the best
/// available test.
fn is_color(m: &ufbx::Material, name: &str) -> bool {
    prop(m, name).is_some_and(|p| p.type_ == ufbx::PropType::Color)
}

/// FBXLoader parseMaterial() + parseParameters().
fn build_material(m: &ufbx::Material, textures: &mut Textures, scene: &mut Scene) -> Material {
    let model = m.shading_model_name.to_ascii_lowercase();
    let mut mat = if model == "lambert" { Material::lambert() } else { Material::phong() };

    if let Some(bump) = prop_real(m, "BumpFactor") {
        mat.bump_scale = bump;
    }
    if let Some(c) = prop_color(m, "Diffuse") {
        mat.color = c;
    } else if is_color(m, "DiffuseColor") {
        if let Some(c) = prop_color(m, "DiffuseColor") {
            mat.color = c;
        }
    }
    let mut emissive = None;
    if let Some(c) = prop_color(m, "Emissive") {
        emissive = Some(c);
    } else if is_color(m, "EmissiveColor") {
        emissive = prop_color(m, "EmissiveColor");
    }
    let emissive_intensity = prop_real(m, "EmissiveFactor").unwrap_or(1.0);
    if let Some(e) = emissive {
        mat.emissive = e * emissive_intensity;
    }

    // Blender-style transparency: 1 - TransparencyFactor, unless that is 0
    // or 1, then Opacity, then opaque.
    let mut opacity = 1.0 - prop_real(m, "TransparencyFactor").unwrap_or(0.0);
    if opacity == 1.0 || opacity == 0.0 {
        opacity = prop_real(m, "Opacity").unwrap_or(1.0);
    }
    mat.opacity = opacity;
    if opacity < 1.0 {
        mat.alpha_mode = AlphaMode::Blend;
    }
    if let Some(s) = prop_real(m, "Shininess") {
        mat.shininess = s;
    }
    if let Some(c) = prop_color(m, "Specular") {
        mat.specular = c;
    } else if is_color(m, "SpecularColor") {
        if let Some(c) = prop_color(m, "SpecularColor") {
            mat.specular = c;
        }
    }

    for t in m.textures.iter() {
        let tex = texture(&t.texture, textures, scene);
        match &*t.material_prop {
            "Bump" => mat.bump_map = tex,
            "Maya|TEX_ao_map" => mat.ao_map = tex,
            "DiffuseColor" | "Maya|TEX_color_map" => mat.map = tex,
            "EmissiveColor" => mat.emissive_map = tex,
            "NormalMap" | "Maya|TEX_normal_map" => mat.normal_map = tex,
            "SpecularColor" => mat.specular_map = tex,
            "TransparentColor" | "TransparencyFactor" => {
                mat.alpha_map = tex;
                mat.alpha_mode = AlphaMode::Blend;
            }
            _ => {}
        }
    }
    mat
}

/// FBXLoader parseTexture(): embedded content wins, else the file name
/// relative to the model; wrap modes and the Scaling/Translation
/// properties become repeat and offset.
fn texture(t: &ufbx::Texture, textures: &mut Textures, scene: &mut Scene) -> Option<TextureRef> {
    // Layered textures: the first layer.
    let t: &ufbx::Texture = match t.layers.first() {
        Some(layer) if t.type_ == ufbx::TextureType::Layered => &layer.texture,
        _ => t,
    };
    let key = format!("fbx-texture-{}", t.element.element_id);
    let image = if !t.content.is_empty() {
        textures.load_bytes(scene, &key, &t.content, Missing::Black)
    } else {
        let content = t.video.as_ref().map(|v| &v.content).filter(|c| !c.is_empty());
        match content {
            Some(bytes) => textures.load_bytes(scene, &key, bytes, Missing::Black),
            None => {
                // RelativeFilename || Filename, appended to the model directory.
                let rel = if !t.relative_filename.is_empty() { &*t.relative_filename } else { &*t.filename };
                textures.load(scene, rel, false)
            }
        }
    }?;
    let mut r = TextureRef::new(image);
    // FBXLoader: WrapModeU/V 0 = repeat, else clamp.
    r.wrap_u = if t.wrap_u == ufbx::WrapMode::Repeat { Wrap::Repeat } else { Wrap::Clamp };
    r.wrap_v = if t.wrap_v == ufbx::WrapMode::Repeat { Wrap::Repeat } else { Wrap::Clamp };
    let props = &t.element.props.props;
    let get = |name: &str| props.iter().find(|p| &*p.name == name).map(|p| Vec2::new(p.value_vec4.x as f32, p.value_vec4.y as f32));
    let repeat = get("Scaling").unwrap_or(Vec2::ONE);
    let offset = get("Translation").unwrap_or(Vec2::ZERO);
    if repeat != Vec2::ONE || offset != Vec2::ZERO {
        r.transform = UvTransform::three(offset, repeat, 0.0, Vec2::ZERO, true);
    }
    Some(r)
}
