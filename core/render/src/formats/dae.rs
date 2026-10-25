// Collada (.dae): what three.js r186's ColladaLoader (ColladaParser +
// ColladaComposer) shows for a file in its rest pose.
//
// The port follows three's object graph closely because many of its quirks
// change the picture: which node gets which matrix, which library entry wins
// a duplicate id, how polygons are triangulated, how skinned meshes are posed
// and even how their bounding box (and therefore the camera framing) comes
// out. Where three throws on a broken file (and the viewer shows an error)
// this loader skips the broken part instead; those spots are marked
// "lenient".
//
// Not drawn, as in three: cameras, lights, animations, kinematics, physics,
// <convex_mesh>/<spline>/<brep> geometry, morph targets, <ph> holes. Vertex
// colours are ignored because three never sets material.vertexColors for
// Collada.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use glam::{DMat3, DMat4, DQuat, DVec2, DVec3, Vec2};
use roxmltree::{Document, Node, NodeId, ParsingOptions};

use super::{invalid, LoadError};
use crate::scene::{
    flip_v, srgb3, Aabb, AlphaMode, Group, Material, Mesh, Placement, Scene, Shading, Side, TextureRef, Topology,
    UvTransform, Wrap,
};
use crate::texture::Textures;

/// Nesting limit for node / instance_node / skeleton recursion; three would
/// overflow its stack on a cyclic reference.
const MAX_DEPTH: usize = 64;
/// Scene-graph objects and emitted vertices, against instance_node trees that
/// multiply out exponentially.
const MAX_OBJECTS: usize = 1 << 20;
const MAX_VERTICES: usize = 1 << 25;
/// Polygons with more corners are fanned instead of ear-clipped, which keeps
/// earcut's O(n^2) work and recursion bounded.
const MAX_EARCUT_CORNERS: usize = 512;
/// An index that is not a valid non-negative integer.
const BAD_INDEX: u32 = u32::MAX;

pub fn load(path: &Path, textures: &mut Textures) -> Result<Scene, LoadError> {
    let bytes = std::fs::read(path)?;
    let source = String::from_utf8_lossy(&bytes);
    let source = source.trim_start_matches('\u{feff}');
    if source.is_empty() {
        // ColladaLoader.parse returns an empty Scene for an empty file.
        return Ok(Scene::default());
    }
    let options = ParsingOptions { allow_dtd: true, ..ParsingOptions::default() };
    let doc = Document::parse_with_options(source, options).map_err(|e| invalid(format!("dae: {e}")))?;
    let collada = doc.root_element();
    if collada.tag_name().name() != "COLLADA" {
        return Err(invalid("dae: no <COLLADA> root element"));
    }

    let lib = Library::new(collada);
    let mut b = Builder {
        lib,
        textures,
        scene: Scene::default(),
        objs: Vec::new(),
        node_cache: HashMap::new(),
        geometry_cache: HashMap::new(),
        source_cache: HashMap::new(),
        skin_cache: HashMap::new(),
        material_cache: HashMap::new(),
        line_material_cache: HashMap::new(),
        fallback_material: None,
        default_mesh_material: None,
        default_line_material: None,
    };

    // ColladaLoader.parse: scene.rotation.set(-PI / 2, 0, 0) for Z_UP (the
    // text is compared as is, untrimmed) and scene.scale *= unit; Object3D
    // composes both into its matrix.
    let asset = child(collada, "asset");
    let unit = asset
        .and_then(|a| child(a, "unit"))
        .and_then(|u| u.attribute("meter"))
        .map(js_float)
        .unwrap_or(1.0);
    let z_up = asset.and_then(|a| child(a, "up_axis")).is_some_and(|u| text(u) == "Z_UP");
    let rotation = if z_up { DQuat::from_xyzw((-std::f64::consts::FRAC_PI_4).sin(), 0.0, 0.0, std::f64::consts::FRAC_PI_4.cos()) } else { DQuat::IDENTITY };
    let root = b.new_obj(Kind::Group);
    b.objs[root].local = compose(DVec3::ZERO, rotation, DVec3::splat(unit));

    // ColladaComposer.parseScene: the visual scene <scene> instantiates.
    // Lenient: without a usable reference the first visual scene is shown.
    let scene_ref = child(collada, "scene")
        .and_then(|s| child(s, "instance_visual_scene"))
        .and_then(|i| i.attribute("url"))
        .and_then(|url| b.lib.visual_scenes.get(parse_id(url)).copied());
    if let Some(visual_scene) = scene_ref.or(b.lib.first_visual_scene) {
        for node in children(visual_scene, "node") {
            if let Some(obj) = b.resolve_node(node).and_then(|n| b.build_node(n, 1)) {
                b.add(root, obj);
            }
        }
    }

    b.emit(root)?;
    Ok(b.scene)
}

// ---------------------------------------------------------------------------
// XML helpers

fn children<'a, 'i>(n: Node<'a, 'i>, name: &'static str) -> impl DoubleEndedIterator<Item = Node<'a, 'i>> {
    n.children().filter(move |c| c.is_element() && c.tag_name().name() == name)
}

fn child<'a, 'i>(n: Node<'a, 'i>, name: &'static str) -> Option<Node<'a, 'i>> {
    children(n, name).next()
}

/// ColladaParser reads child lists in a loop that overwrites, so the last
/// element of a kind wins.
fn last_child<'a, 'i>(n: Node<'a, 'i>, name: &'static str) -> Option<Node<'a, 'i>> {
    children(n, name).next_back()
}

/// DOM textContent.
fn text<'a>(n: Node<'a, '_>) -> Cow<'a, str> {
    let mut it = n.children();
    match (it.next(), it.next()) {
        (None, _) => Cow::Borrowed(""),
        (Some(c), None) if c.is_text() => Cow::Borrowed(c.text().unwrap_or("")),
        _ => Cow::Owned(n.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect()),
    }
}

/// ColladaParser's parseId: drops the first character (the '#').
fn parse_id(s: &str) -> &str {
    s.char_indices().nth(1).map_or("", |(i, _)| &s[i..])
}

/// JavaScript parseFloat: the longest numeric prefix, NaN if there is none
/// (so "1.#QNAN" is 1, "inf" is NaN).
fn js_float(token: &str) -> f64 {
    let s = token.trim_start();
    let b = s.as_bytes();
    let sign = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    if matches!(b.get(sign), Some(c) if c.is_ascii_digit() || *c == b'.') {
        if let Ok(v) = s.parse::<f64>() {
            return v;
        }
    }
    if s[sign..].starts_with("Infinity") {
        return if b.first() == Some(&b'-') { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    let digits = |mut i: usize| {
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        i
    };
    let int_end = digits(sign);
    let mut end = int_end;
    let mut mantissa = int_end > sign;
    if b.get(end) == Some(&b'.') {
        let frac_end = digits(end + 1);
        mantissa |= frac_end > end + 1;
        end = frac_end;
    }
    if !mantissa {
        return f64::NAN;
    }
    if matches!(b.get(end), Some(b'e' | b'E')) {
        let exp_sign = end + 1 + usize::from(matches!(b.get(end + 1), Some(b'+' | b'-')));
        let exp_end = digits(exp_sign);
        if exp_end > exp_sign {
            end = exp_end;
        }
    }
    s[..end].parse::<f64>().unwrap_or(f64::NAN)
}

/// JavaScript parseInt (base 10): the leading integer, None for NaN.
fn js_int(token: &str) -> Option<i64> {
    if let Ok(v) = token.parse::<i64>() {
        return Some(v);
    }
    let s = token.trim_start();
    let b = s.as_bytes();
    let sign = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let mut end = sign;
    while end < b.len() && b[end].is_ascii_digit() {
        end += 1;
    }
    if end == sign {
        return None;
    }
    // Saturate absurdly long digit runs instead of failing.
    Some(s[..end].parse::<i64>().unwrap_or(if b[0] == b'-' { i64::MIN } else { i64::MAX }))
}

fn floats_f32(s: &str) -> Vec<f32> {
    s.split_ascii_whitespace().map(|t| js_float(t) as f32).collect()
}

fn floats_f64(s: &str) -> Vec<f64> {
    s.split_ascii_whitespace().map(js_float).collect()
}

/// Index lists; anything that is not a usable index becomes BAD_INDEX.
fn indices(s: &str) -> Vec<u32> {
    s.split_ascii_whitespace()
        .map(|t| js_int(t).and_then(|v| u32::try_from(v).ok()).unwrap_or(BAD_INDEX))
        .collect()
}

/// Matrix4.fromArray(array, offset).transpose(): Collada stores row-major.
fn row_major(values: &[f64], offset: usize) -> Option<DMat4> {
    let v = values.get(offset..offset.checked_add(16)?)?;
    let mut a = [0.0; 16];
    a.copy_from_slice(v);
    Some(DMat4::from_cols_array(&a).transpose())
}

// ---------------------------------------------------------------------------
// three's Matrix4 / Quaternion maths, where it differs from glam

/// Matrix4.compose.
fn compose(p: DVec3, q: DQuat, s: DVec3) -> DMat4 {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    DMat4::from_cols_array(&[
        (1.0 - (yy + zz)) * s.x,
        (xy + wz) * s.x,
        (xz - wy) * s.x,
        0.0,
        (xy - wz) * s.y,
        (1.0 - (xx + zz)) * s.y,
        (yz + wx) * s.y,
        0.0,
        (xz + wy) * s.z,
        (yz - wx) * s.z,
        (1.0 - (xx + yy)) * s.z,
        0.0,
        p.x,
        p.y,
        p.z,
        1.0,
    ])
}

/// Matrix4.decompose: shear and the projective row are lost, and a
/// singular matrix keeps only its translation.
fn decompose(m: &DMat4) -> (DVec3, DQuat, DVec3) {
    let position = m.w_axis.truncate();
    let det = DMat3::from_mat4(*m).determinant();
    if det == 0.0 {
        return (position, DQuat::IDENTITY, DVec3::ONE);
    }
    let mut sx = m.x_axis.truncate().length();
    let sy = m.y_axis.truncate().length();
    let sz = m.z_axis.truncate().length();
    if det < 0.0 {
        sx = -sx;
    }
    let (c0, c1, c2) = (m.x_axis.truncate() / sx, m.y_axis.truncate() / sy, m.z_axis.truncate() / sz);
    // Quaternion.setFromRotationMatrix; mRC = row R, column C.
    let (m11, m12, m13) = (c0.x, c1.x, c2.x);
    let (m21, m22, m23) = (c0.y, c1.y, c2.y);
    let (m31, m32, m33) = (c0.z, c1.z, c2.z);
    let trace = m11 + m22 + m33;
    let q = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        DQuat::from_xyzw((m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s)
    } else if m11 > m22 && m11 > m33 {
        let s = 2.0 * (1.0 + m11 - m22 - m33).sqrt();
        DQuat::from_xyzw(0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s)
    } else if m22 > m33 {
        let s = 2.0 * (1.0 + m22 - m11 - m33).sqrt();
        DQuat::from_xyzw((m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s)
    } else {
        let s = 2.0 * (1.0 + m33 - m11 - m22).sqrt();
        DQuat::from_xyzw((m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s)
    };
    (position, q, DVec3::new(sx, sy, sz))
}

/// What Object3D ends up with after `matrix.copy(m); matrix.decompose(...)`
/// and the next updateMatrix().
fn recompose(m: &DMat4) -> DMat4 {
    let (p, q, s) = decompose(m);
    let r = compose(p, q, s);
    if r.is_finite() {
        r
    } else {
        *m
    }
}

/// Matrix4.makeRotationAxis, which assumes (but does not enforce) a unit
/// axis.
fn rotation_axis(axis: DVec3, angle: f64) -> DMat4 {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    let (tx, ty) = (t * x, t * y);
    DMat4::from_cols_array(&[
        tx * x + c,
        tx * y + s * z,
        tx * z - s * y,
        0.0,
        tx * y - s * z,
        ty * y + c,
        ty * z + s * x,
        0.0,
        tx * z + s * y,
        ty * z - s * x,
        t * z * z + c,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}

/// Quaternion.setFromAxisAngle, again without normalizing the axis.
fn quat_axis_angle(axis: DVec3, angle: f64) -> DQuat {
    let (s, c) = (angle / 2.0).sin_cos();
    DQuat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c)
}

fn vec3_of(v: &[f64]) -> Option<DVec3> {
    Some(DVec3::new(*v.first()?, *v.get(1)?, *v.get(2)?))
}

// ---------------------------------------------------------------------------
// Libraries

/// The id maps ColladaParser builds. It only reads the first
/// <library_xxx> of each kind; within it a later duplicate id replaces an
/// earlier one, except for nodes, where the first registered one stays.
struct Library<'a, 'i> {
    images: HashMap<&'a str, Node<'a, 'i>>,
    effects: HashMap<&'a str, Node<'a, 'i>>,
    materials: HashMap<&'a str, Node<'a, 'i>>,
    geometries: HashMap<&'a str, Node<'a, 'i>>,
    controllers: HashMap<&'a str, Node<'a, 'i>>,
    cameras: HashMap<&'a str, Node<'a, 'i>>,
    lights: HashMap<&'a str, Node<'a, 'i>>,
    visual_scenes: HashMap<&'a str, Node<'a, 'i>>,
    first_visual_scene: Option<Node<'a, 'i>>,
    nodes: HashMap<&'a str, Node<'a, 'i>>,
    /// geometry id -> the <skin> whose weights three attaches to it: every
    /// controller is built up front and overwrites the geometry's skin
    /// sources, so the last controller in the library wins.
    geometry_skins: HashMap<&'a str, Node<'a, 'i>>,
}

impl<'a, 'i> Library<'a, 'i> {
    fn new(collada: Node<'a, 'i>) -> Library<'a, 'i> {
        let section = |name: &'static str, item: &'static str| {
            let mut map = HashMap::new();
            if let Some(lib) = child(collada, name) {
                for e in children(lib, item) {
                    if let Some(id) = e.attribute("id") {
                        map.insert(id, e);
                    }
                }
            }
            map
        };
        let mut geometry_skins = HashMap::new();
        if let Some(lib) = child(collada, "library_controllers") {
            for controller in children(lib, "controller") {
                if let Some(skin) = last_child(controller, "skin") {
                    if let Some(source) = skin.attribute("source") {
                        geometry_skins.insert(parse_id(source), skin);
                    }
                }
            }
        }
        // Nodes register children before parents (parseNode recurses
        // first), library_nodes before the visual scenes.
        let mut nodes = HashMap::new();
        let first_visual_scene_lib = child(collada, "library_visual_scenes");
        let roots = child(collada, "library_nodes")
            .into_iter()
            .chain(first_visual_scene_lib.into_iter().flat_map(|l| children(l, "visual_scene")));
        for root in roots {
            for top in children(root, "node") {
                register_nodes(top, &mut nodes);
            }
        }
        Library {
            images: section("library_images", "image"),
            effects: section("library_effects", "effect"),
            materials: section("library_materials", "material"),
            geometries: section("library_geometries", "geometry"),
            controllers: section("library_controllers", "controller"),
            cameras: section("library_cameras", "camera"),
            lights: section("library_lights", "light"),
            visual_scenes: section("library_visual_scenes", "visual_scene"),
            first_visual_scene: first_visual_scene_lib.and_then(|l| child(l, "visual_scene")),
            nodes,
            geometry_skins,
        }
    }
}

/// Post-order registration without recursion (files can nest deeply).
fn register_nodes<'a, 'i>(top: Node<'a, 'i>, nodes: &mut HashMap<&'a str, Node<'a, 'i>>) {
    let mut stack = vec![(top, false)];
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            if let Some(id) = node.attribute("id") {
                nodes.entry(id).or_insert(node);
            }
        } else {
            stack.push((node, true));
            let kids: Vec<_> = children(node, "node").collect();
            stack.extend(kids.into_iter().rev().map(|k| (k, false)));
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry

/// A <source>'s float array and accessor stride.
struct Source {
    array: Vec<f32>,
    stride: usize,
}

impl Source {
    fn parse(node: Node) -> Option<Source> {
        let mut array = Vec::new();
        let mut stride = 3;
        for c in node.children().filter(|c| c.is_element()) {
            match c.tag_name().name() {
                "float_array" => array = floats_f32(&text(c)),
                "technique_common" => {
                    if let Some(accessor) = child(c, "accessor") {
                        // Lenient: a missing stride is NaN in three; the
                        // specification's default is 1.
                        stride = accessor.attribute("stride").and_then(js_int).unwrap_or(1).max(0) as usize;
                    }
                }
                _ => {}
            }
        }
        (stride > 0).then_some(Source { array, stride })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PartKind {
    Triangles,
    /// LineSegments.
    Lines,
    /// Line: one strip through all vertices (of a draw range).
    LineStrip,
}

/// One primitive element (<triangles>, <polylist>, ...).
struct Primitive<'a> {
    material: Option<&'a str>,
    count: f64,
    /// (semantic + set, source id, offset), in first-insertion order.
    inputs: Vec<(String, &'a str, usize)>,
    stride: usize,
    has_uv: bool,
    vcount: Option<Vec<u32>>,
    p: Vec<u32>,
}

impl<'a> Primitive<'a> {
    fn parse(node: Node<'a, '_>) -> Primitive<'a> {
        let kind = node.tag_name().name();
        let mut prim = Primitive {
            material: node.attribute("material"),
            count: node.attribute("count").and_then(js_int).map_or(f64::NAN, |c| c as f64),
            inputs: Vec::new(),
            stride: 0,
            has_uv: false,
            vcount: None,
            p: Vec::new(),
        };
        for c in node.children().filter(|c| c.is_element()) {
            match c.tag_name().name() {
                "input" => {
                    let semantic = c.attribute("semantic").unwrap_or("");
                    let source = parse_id(c.attribute("source").unwrap_or(""));
                    // Lenient: three's stride turns NaN without an offset.
                    let offset = c.attribute("offset").and_then(js_int).unwrap_or(0).clamp(0, 1 << 16) as usize;
                    let set = c.attribute("set").and_then(js_int).unwrap_or(0);
                    let name = if set > 0 { format!("{semantic}{set}") } else { semantic.to_string() };
                    match prim.inputs.iter_mut().find(|(n, _, _)| *n == name) {
                        Some(input) => *input = (name, source, offset),
                        None => prim.inputs.push((name, source, offset)),
                    }
                    prim.stride = prim.stride.max(offset + 1);
                    if semantic == "TEXCOORD" {
                        prim.has_uv = true;
                    }
                }
                "vcount" => prim.vcount = Some(indices(&text(c)).into_iter().map(|v| if v == BAD_INDEX { 0 } else { v }).collect()),
                // Only the last <p> survives - for <polygons>, which has one
                // <p> per polygon, three thus draws just the last polygon.
                "p" => prim.p = indices(&text(c)),
                _ => {}
            }
        }
        if kind == "polygons" {
            prim.vcount = Some(prim.p.len().checked_div(prim.stride).map(|n| n as u32).into_iter().collect());
        }
        prim
    }

    /// The group size buildGeometryType declares; NaN where three's
    /// arithmetic produces NaN.
    fn declared_count(&self, kind: &str) -> f64 {
        match kind {
            "lines" | "linestrips" => self.count * 2.0,
            "triangles" => self.count * 3.0,
            _ => {
                let vcount = self.vcount.as_deref().unwrap_or(&[]);
                let mut total = 0.0;
                let mut g = 0usize;
                // `for (g = 0; g < count; g++)`, reading past vcount gives NaN.
                while (g as f64) < self.count {
                    match vcount.get(g) {
                        Some(3) => total += 3.0,
                        Some(4) => total += 6.0,
                        Some(&v) => total += (v as f64 - 2.0) * 3.0,
                        None => return f64::NAN,
                    }
                    g += 1;
                }
                total
            }
        }
    }

    /// ColladaComposer.buildGeometryData: the source items one input
    /// contributes, as a flat array with the source's stride. Lenient: where
    /// three would push nothing or NaN for a bad index, a zero item keeps the
    /// attributes aligned.
    fn gather(&self, source: &Source, offset: usize) -> Vec<f32> {
        let ss = source.stride;
        let zeros = vec![0f32; ss];
        let p = &self.p;
        let mut out = Vec::new();
        let push = |i: usize, out: &mut Vec<f32>| {
            let item = p
                .get(i + offset)
                .filter(|&&v| v != BAD_INDEX)
                .and_then(|&v| source.array.get(v as usize * ss..v as usize * ss + ss));
            out.extend_from_slice(item.unwrap_or(&zeros));
        };
        let stride = self.stride;
        match &self.vcount {
            None => {
                out.reserve(p.len() / stride * ss);
                for i in (0..p.len()).step_by(stride) {
                    push(i, &mut out);
                }
            }
            Some(vcount) => {
                let mut index = 0usize;
                for &count in vcount {
                    let count = count as usize;
                    let end = index.saturating_add(stride.saturating_mul(count));
                    // Lenient: polygons running past <p> are skipped.
                    if end > p.len() {
                        break;
                    }
                    match count {
                        3 => {
                            for k in 0..3 {
                                push(index + stride * k, &mut out);
                            }
                        }
                        4 => {
                            for k in [0, 1, 3, 1, 2, 3] {
                                push(index + stride * k, &mut out);
                            }
                        }
                        n if n > 4 => {
                            for k in self.triangulate(source, index, n) {
                                push(index + stride * k, &mut out);
                            }
                        }
                        _ => {}
                    }
                    index = end;
                }
            }
        }
        out
    }

    /// The >4-corner branch of buildGeometryData: project the polygon on the
    /// dominant plane of its first triangle and run ShapeUtils'
    /// triangulation. Faithful to three, the contour is read from the input
    /// being gathered (with the offset-0 index), so attributes of concave
    /// polygons can be triangulated differently from the positions.
    fn triangulate(&self, source: &Source, index: usize, count: usize) -> Vec<usize> {
        if count > MAX_EARCUT_CORNERS {
            return (1..count - 1).flat_map(|k| [0, k, k + 1]).collect();
        }
        // Reads past the source are `undefined`, which `new Vector3()`
        // replaces with its default 0.
        let at = |i: usize| source.array.get(i).map_or(0.0, |&v| v as f64);
        let corners: Vec<DVec3> = (0..count)
            .map(|k| {
                let v = self.p[index + self.stride * k];
                let base = if v == BAD_INDEX { usize::MAX - 2 } else { v as usize * source.stride };
                DVec3::new(at(base), at(base.saturating_add(1)), at(base.saturating_add(2)))
            })
            .collect();
        // Triangle.getNormal
        let normal = (corners[2] - corners[1]).cross(corners[0] - corners[1]);
        let normal = if normal.length_squared() > 0.0 { normal.normalize() } else { DVec3::ZERO };
        let a = normal.abs();
        let mut flat: Vec<DVec2> = corners
            .iter()
            .map(|v| {
                if a.x > a.y && a.x > a.z {
                    DVec2::new(v.y, v.z)
                } else if a.y > a.z {
                    DVec2::new(v.x, v.z)
                } else {
                    DVec2::new(v.x, v.y)
                }
            })
            .collect();
        let clockwise = shape_area(&flat) < 0.0;
        if clockwise {
            flat.reverse();
        }
        // ShapeUtils.triangulateShape's removeDupEndPts.
        if flat.len() > 2 && flat[flat.len() - 1] == flat[0] {
            flat.pop();
        }
        let faces = earcut::triangulate(&flat);
        let mut out = Vec::with_capacity(faces.len());
        for f in faces.as_chunks::<3>().0 {
            if clockwise {
                out.extend([count - 1 - f[0], count - 1 - f[2], count - 1 - f[1]]);
            } else {
                out.extend([f[0], f[1], f[2]]);
            }
        }
        out
    }
}

/// ShapeUtils.area.
fn shape_area(contour: &[DVec2]) -> f64 {
    let n = contour.len();
    let mut a = 0.0;
    let mut p = n.wrapping_sub(1);
    for q in 0..n {
        a += contour[p].x * contour[q].y - contour[q].x * contour[p].y;
        p = q;
    }
    a * 0.5
}

/// Per-vertex skin data three attaches to a geometry (buildSkin):
/// the four heaviest joints, weights normalized (normalizeSkinWeights).
type SkinVertex = ([u32; 4], [f32; 4]);

/// One BufferGeometry of buildGeometryType: all primitives of one type.
struct Part {
    kind: PartKind,
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    uv0: Option<Vec<[f32; 2]>>,
    uv1: Option<Vec<[f32; 2]>>,
    skin: Option<Vec<SkinVertex>>,
    /// geometry.addGroup(start, count, primitiveIndex) as declared.
    groups: Vec<(f64, f64)>,
    material_keys: Vec<String>,
}

fn items3(flat: &[f32], stride: usize, out: &mut Vec<[f32; 3]>) {
    out.extend(flat.chunks_exact(stride).map(|c| {
        [c[0], c.get(1).copied().unwrap_or(0.0), c.get(2).copied().unwrap_or(0.0)]
    }));
}

fn items2(flat: &[f32], stride: usize, out: &mut Vec<[f32; 2]>) {
    out.extend(flat.chunks_exact(stride).map(|c| flip_v([c[0], c.get(1).copied().unwrap_or(0.0)])));
}

/// Attributes shorter or longer than the positions are padded / cut, so
/// one primitive without normals cannot misalign the rest.
fn fit<T: Copy>(v: Vec<T>, n: usize, zero: T) -> Option<Vec<T>> {
    if v.is_empty() {
        return None;
    }
    let mut v = v;
    v.resize(n, zero);
    Some(v)
}

/// A controller's <skin>.
struct Skin {
    bind_shape: DMat4,
    /// Joint names (Name_array) with their inverse bind matrices.
    joints: Vec<(String, DMat4)>,
    /// buildSkin's per-vertex index/weight lists (stride 4), indexed by the
    /// position-source vertex.
    vertex_indices: Option<Source>,
    vertex_weights: Option<Source>,
}

/// A skin <input>: semantic, source id, offset.
type Input<'a> = (&'a str, &'a str, i64);

fn skin_inputs<'a>(n: Node<'a, '_>) -> Vec<Input<'a>> {
    children(n, "input")
        .map(|i| {
            let semantic = i.attribute("semantic").unwrap_or("");
            let source = parse_id(i.attribute("source").unwrap_or(""));
            (semantic, source, i.attribute("offset").and_then(js_int).unwrap_or(0))
        })
        .collect()
}

/// ColladaParser keys inputs by semantic, so the last one wins.
fn find_input<'a>(list: &[Input<'a>], semantic: &str) -> Option<(&'a str, i64)> {
    list.iter().rev().find(|(s, _, _)| *s == semantic).map(|&(_, id, offset)| (id, offset))
}

impl Skin {
    fn parse(node: Node) -> Skin {
        let sources: HashMap<&str, Node> =
            children(node, "source").filter_map(|s| Some((s.attribute("id")?, s))).collect();
        let bind_shape = last_child(node, "bind_shape_matrix")
            .and_then(|m| row_major(&floats_f64(&text(m)), 0))
            .unwrap_or(DMat4::IDENTITY);
        // Joints: names and inverse bind matrices. Lenient where three
        // throws: a missing matrix is the identity.
        let joint_inputs = last_child(node, "joints").map(skin_inputs).unwrap_or_default();
        let names: Vec<String> = find_input(&joint_inputs, "JOINT")
            .and_then(|(id, _)| sources.get(id).copied())
            .and_then(|s| last_child(s, "Name_array"))
            .map(|a| text(a).split_ascii_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        let inverse = find_input(&joint_inputs, "INV_BIND_MATRIX").and_then(|(id, _)| sources.get(id).copied());
        let (inverse_values, inverse_stride) = inverse.map_or((Vec::new(), 16), |s| {
            let values = last_child(s, "float_array").map(|a| floats_f64(&text(a))).unwrap_or_default();
            let stride = child(s, "technique_common")
                .and_then(|t| child(t, "accessor"))
                .and_then(|a| a.attribute("stride"))
                .and_then(js_int)
                .map_or(16, |v| v.max(0) as usize);
            (values, stride)
        });
        let joints = names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, row_major(&inverse_values, i * inverse_stride).unwrap_or(DMat4::IDENTITY)))
            .collect();

        let (vertex_indices, vertex_weights) = Skin::parse_weights(node, &sources).unzip();
        Skin { bind_shape, joints, vertex_indices, vertex_weights }
    }

    /// buildSkin's loop over <vertex_weights>: per vertex, the (joint,
    /// weight) pairs sorted by descending weight (stable, like Array.sort)
    /// and cut or padded to four. Its `stride += 2` assumes exactly two
    /// inputs, as three does.
    fn parse_weights(
        node: Node,
        sources: &HashMap<&str, Node>,
    ) -> Option<(Source, Source)> {
        let vw = last_child(node, "vertex_weights")?;
        let list = skin_inputs(vw);
        let (_, joint_offset) = find_input(&list, "JOINT")?;
        let (weight_id, weight_offset) = find_input(&list, "WEIGHT")?;
        let weights = last_child(*sources.get(weight_id)?, "float_array").map(|a| floats_f32(&text(a)))?;
        let vcount = indices(&text(last_child(vw, "vcount")?));
        let v: Vec<i64> = last_child(vw, "v")
            .map(|v| text(v).split_ascii_whitespace().map(|t| js_int(t).unwrap_or(-1)).collect())
            .unwrap_or_default();
        let get = |i: i64| usize::try_from(i).ok().and_then(|i| v.get(i)).copied().unwrap_or(-1);
        let mut idx = Vec::with_capacity(vcount.len() * 4);
        let mut wts = Vec::with_capacity(vcount.len() * 4);
        let mut stride: i64 = 0;
        let mut pairs: Vec<(f32, f32)> = Vec::new();
        for &count in &vcount {
            let count = if count == BAD_INDEX { 0 } else { count as usize };
            // A count beyond the remaining <v> data only yields padding.
            let count = count.min(v.len() / 2 + 1);
            pairs.clear();
            for _ in 0..count {
                let joint = get(stride.saturating_add(joint_offset));
                let weight_index = get(stride.saturating_add(weight_offset));
                let weight = usize::try_from(weight_index).ok().and_then(|i| weights.get(i)).copied().unwrap_or(0.0);
                let weight = if weight.is_nan() { 0.0 } else { weight };
                pairs.push((joint.max(0) as f32, weight));
                stride += 2;
            }
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            for j in 0..4 {
                let (i, w) = pairs.get(j).copied().unwrap_or((0.0, 0.0));
                idx.push(i);
                wts.push(w);
            }
        }
        Some((Source { array: idx, stride: 4 }, Source { array: wts, stride: 4 }))
    }
}

// ---------------------------------------------------------------------------
// The object graph three builds

enum Kind {
    Group,
    Bone,
    /// Camera or light: only matters for how a node's children collapse.
    Other,
    Drawable(Rc<Drawable>),
}

struct Drawable {
    part: Rc<Part>,
    /// Scene material per material slot; one entry means three's single
    /// material (groups ignored).
    materials: Vec<usize>,
    skin: Option<Rc<SkinBinding>>,
}

/// SkinnedMesh.bind(skeleton, bindShapeMatrix).
struct SkinBinding {
    bind_shape: DMat4,
    /// skeleton.bones / boneInverses. None: a joint three could not place
    /// (it throws there); lenient, it stays in its bind pose.
    bones: Vec<Option<(usize, DMat4)>>,
}

struct Obj {
    kind: Kind,
    /// Bones are matched to joints by this name (the node's sid).
    name: Option<String>,
    local: DMat4,
    parent: Option<usize>,
    children: Vec<usize>,
}

struct Builder<'a, 'i, 't> {
    lib: Library<'a, 'i>,
    textures: &'t mut Textures,
    scene: Scene,
    objs: Vec<Obj>,
    /// getNode's cache; None while a node is being built (a cycle) or when
    /// it failed.
    node_cache: HashMap<NodeId, Option<usize>>,
    geometry_cache: HashMap<NodeId, Rc<Vec<Rc<Part>>>>,
    source_cache: HashMap<NodeId, Option<Rc<Source>>>,
    skin_cache: HashMap<NodeId, Rc<Skin>>,
    material_cache: HashMap<NodeId, usize>,
    line_material_cache: HashMap<usize, usize>,
    fallback_material: Option<usize>,
    default_mesh_material: Option<usize>,
    default_line_material: Option<usize>,
}

impl<'a, 'i, 't> Builder<'a, 'i, 't> {
    fn new_obj(&mut self, kind: Kind) -> usize {
        self.objs.push(Obj { kind, name: None, local: DMat4::IDENTITY, parent: None, children: Vec::new() });
        self.objs.len() - 1
    }

    /// Object3D.add: an object has one parent, adding it elsewhere moves it.
    fn add(&mut self, parent: usize, obj: usize) {
        if let Some(old) = self.objs[obj].parent {
            self.objs[old].children.retain(|&c| c != obj);
        }
        self.objs[obj].parent = Some(parent);
        self.objs[parent].children.push(obj);
    }

    /// Object3D.clone(): a deep copy; meshes share geometry, materials and
    /// (for skinned meshes) the original skeleton.
    fn clone_obj(&mut self, src: usize) -> Option<usize> {
        let root = self.copy_obj(src);
        let mut stack = vec![(src, root)];
        while let Some((s, d)) = stack.pop() {
            if self.objs.len() > MAX_OBJECTS {
                return None;
            }
            for c in self.objs[s].children.clone() {
                let copy = self.copy_obj(c);
                self.add(d, copy);
                stack.push((c, copy));
            }
        }
        Some(root)
    }

    fn copy_obj(&mut self, src: usize) -> usize {
        let o = &self.objs[src];
        let kind = match &o.kind {
            Kind::Group => Kind::Group,
            Kind::Bone => Kind::Bone,
            Kind::Other => Kind::Other,
            Kind::Drawable(d) => Kind::Drawable(d.clone()),
        };
        let (name, local) = (o.name.clone(), o.local);
        let id = self.new_obj(kind);
        self.objs[id].name = name;
        self.objs[id].local = local;
        id
    }

    /// The node an id reference resolves to (library.nodes, first wins);
    /// nodes without an id get a unique one in three, i.e. stand for
    /// themselves.
    fn resolve_node(&self, node: Node<'a, 'i>) -> Option<Node<'a, 'i>> {
        match node.attribute("id") {
            Some(id) => self.lib.nodes.get(id).copied(),
            None => Some(node),
        }
    }

    /// ColladaComposer.buildNode (cached like getNode).
    fn build_node(&mut self, node: Node<'a, 'i>, depth: usize) -> Option<usize> {
        if let Some(cached) = self.node_cache.get(&node.id()) {
            return *cached;
        }
        if depth > MAX_DEPTH || self.objs.len() > MAX_OBJECTS {
            return None;
        }
        self.node_cache.insert(node.id(), None);

        // Children in buildNode's order: nodes, cameras, controllers,
        // lights, geometries, instance_nodes.
        let mut objects = Vec::new();
        let child_nodes: Vec<_> = children(node, "node").collect();
        for c in &child_nodes {
            if let Some(o) = self.resolve_node(*c).and_then(|n| self.build_node(n, depth + 1)) {
                objects.push(o);
            }
        }
        for c in children(node, "instance_camera") {
            if c.attribute("url").is_some_and(|u| self.lib.cameras.contains_key(parse_id(u))) {
                objects.push(self.new_obj(Kind::Other));
            }
        }
        for c in children(node, "instance_controller") {
            let built = self.instance_controller(c, depth);
            objects.extend(built);
        }
        for c in children(node, "instance_light") {
            if c.attribute("url").is_some_and(|u| self.lib.lights.contains_key(parse_id(u))) {
                objects.push(self.new_obj(Kind::Other));
            }
        }
        for c in children(node, "instance_geometry") {
            let built = self.instance_geometry(c);
            objects.extend(built);
        }
        for c in children(node, "instance_node") {
            let target = c.attribute("url").and_then(|u| self.lib.nodes.get(parse_id(u)).copied());
            if let Some(o) = target.and_then(|n| self.build_node(n, depth + 1)) {
                if let Some(copy) = self.clone_obj(o) {
                    objects.push(copy);
                }
            }
        }

        let joint = node.attribute("type") == Some("JOINT");
        // A node with a single instance becomes that object itself, and the
        // node's matrix replaces the object's own (also a cloned
        // instance_node's).
        let obj = if child_nodes.is_empty() && objects.len() == 1 {
            objects[0]
        } else {
            let group = self.new_obj(if joint { Kind::Bone } else { Kind::Group });
            for o in objects {
                self.add(group, o);
            }
            group
        };
        if joint {
            self.objs[obj].name = node.attribute("sid").map(str::to_string);
        }

        let result = if !joint && has_pivot_transforms(node) {
            // wrapWithTransformHierarchy: one Group per transform; the
            // content keeps its own matrix.
            let wrapper = self.new_obj(Kind::Group);
            self.objs[wrapper].local = pivot_chain(node);
            self.add(wrapper, obj);
            wrapper
        } else {
            self.objs[obj].local = recompose(&node_matrix(node));
            obj
        };
        self.node_cache.insert(node.id(), Some(result));
        Some(result)
    }

    fn instance_geometry(&mut self, inst: Node<'a, 'i>) -> Vec<usize> {
        // Lenient: an unknown geometry throws in three.
        let Some(geometry) = inst.attribute("url").and_then(|u| self.lib.geometries.get(parse_id(u)).copied()) else {
            return Vec::new();
        };
        let parts = self.geometry(geometry);
        let bindings = material_bindings(inst);
        parts
            .iter()
            .map(|part| {
                let materials = self.resolve_materials(part, &bindings);
                let d = Drawable { part: part.clone(), materials, skin: None };
                self.new_obj(Kind::Drawable(Rc::new(d)))
            })
            .collect()
    }

    fn instance_controller(&mut self, inst: Node<'a, 'i>, depth: usize) -> Vec<usize> {
        let Some(controller) = inst.attribute("url").and_then(|u| self.lib.controllers.get(parse_id(u)).copied()) else {
            return Vec::new();
        };
        // data.id: the source of the controller's last <skin> or <morph>.
        let Some(geometry_id) = controller
            .children()
            .rfind(|c| c.is_element() && matches!(c.tag_name().name(), "skin" | "morph"))
            .and_then(|c| c.attribute("source"))
            .map(parse_id)
        else {
            return Vec::new();
        };
        let Some(geometry) = self.lib.geometries.get(geometry_id).copied() else {
            return Vec::new();
        };
        let parts = self.geometry(geometry);
        let bindings = material_bindings(inst);
        // Lenient: a morph-only controller (three throws) shows its base
        // geometry unskinned.
        let binding = last_child(controller, "skin").map(|skin| {
            let skin = self.skin(skin);
            let skeletons: Vec<String> =
                children(inst, "skeleton").map(|s| parse_id(text(s).trim()).to_string()).collect();
            Rc::new(self.build_skeleton(&skeletons, &skin, depth))
        });
        parts
            .iter()
            .map(|part| {
                let materials = self.resolve_materials(part, &bindings);
                // SkinnedMesh only if the geometry carries skin attributes.
                let skin = if part.skin.is_some() { binding.clone() } else { None };
                let d = Drawable { part: part.clone(), materials, skin };
                self.new_obj(Kind::Drawable(Rc::new(d)))
            })
            .collect()
    }

    /// buildSkeleton + buildBoneHierarchy.
    fn build_skeleton(&mut self, skeletons: &[String], skin: &Skin, depth: usize) -> SkinBinding {
        let mut roots = Vec::new();
        for id in skeletons {
            if let Some(node) = self.lib.nodes.get(id.as_str()).copied() {
                roots.extend(self.build_node(node, depth + 1));
            } else if let Some(vs) = self.lib.visual_scenes.get(id.as_str()).copied() {
                // #13335: the skeleton names the visual scene; its JOINT
                // children are the roots.
                for c in children(vs, "node").filter(|c| c.attribute("type") == Some("JOINT")) {
                    if let Some(o) = self.resolve_node(c).and_then(|n| self.build_node(n, depth + 1)) {
                        roots.push(o);
                    }
                }
            }
        }
        // root.traverse(): pre-order, collecting Bones.
        let mut bone_data: Vec<(usize, DMat4, bool)> = Vec::new();
        for root in roots {
            let mut stack = vec![root];
            let mut seen = 0usize;
            while let Some(o) = stack.pop() {
                seen += 1;
                if seen > MAX_OBJECTS {
                    break;
                }
                if matches!(self.objs[o].kind, Kind::Bone) {
                    let name = self.objs[o].name.as_deref();
                    let inverse = skin
                        .joints
                        .iter()
                        .find(|(n, _)| Some(n.as_str()) == name)
                        .map_or(DMat4::IDENTITY, |(_, m)| *m);
                    bone_data.push((o, inverse, false));
                }
                stack.extend(self.objs[o].children.iter().rev());
            }
        }
        // Sort by the controller's joint order; JavaScript array semantics:
        // unmatched bones are pushed after the highest matched slot.
        let mut bones: Vec<Option<(usize, DMat4)>> = Vec::new();
        for (i, (name, _)) in skin.joints.iter().enumerate() {
            if let Some(data) = bone_data.iter_mut().find(|(o, _, _)| self.objs[*o].name.as_deref() == Some(name.as_str())) {
                if bones.len() <= i {
                    bones.resize(i + 1, None);
                }
                bones[i] = Some((data.0, data.1));
                data.2 = true;
            }
        }
        bones.extend(bone_data.iter().filter(|d| !d.2).map(|d| Some((d.0, d.1))));
        SkinBinding { bind_shape: skin.bind_shape, bones }
    }

    fn skin(&mut self, node: Node<'a, 'i>) -> Rc<Skin> {
        self.skin_cache.entry(node.id()).or_insert_with(|| Rc::new(Skin::parse(node))).clone()
    }

    fn source(&mut self, node: Node<'a, 'i>) -> Option<Rc<Source>> {
        self.source_cache.entry(node.id()).or_insert_with(|| Source::parse(node).map(Rc::new)).clone()
    }

    /// buildGeometry: one Part per primitive type, in order of appearance.
    fn geometry(&mut self, node: Node<'a, 'i>) -> Rc<Vec<Rc<Part>>> {
        if let Some(cached) = self.geometry_cache.get(&node.id()) {
            return cached.clone();
        }
        let parts = Rc::new(self.build_geometry(node));
        self.geometry_cache.insert(node.id(), parts.clone());
        parts
    }

    fn build_geometry(&mut self, node: Node<'a, 'i>) -> Vec<Rc<Part>> {
        let Some(mesh) = child(node, "mesh") else {
            return Vec::new();
        };
        let sources: HashMap<&str, Node> =
            children(mesh, "source").filter_map(|s| Some((s.attribute("id")?, s))).collect();
        // The last <vertices>: semantic -> source, first-insertion order.
        let mut vertices: Vec<(&str, &str)> = Vec::new();
        if let Some(v) = last_child(mesh, "vertices") {
            for input in v.children().filter(|c| c.is_element()) {
                let semantic = input.attribute("semantic").unwrap_or("");
                let source = parse_id(input.attribute("source").unwrap_or(""));
                match vertices.iter_mut().find(|(s, _)| *s == semantic) {
                    Some(entry) => entry.1 = source,
                    None => vertices.push((semantic, source)),
                }
            }
        }
        let mut by_type: Vec<(&str, Vec<Primitive>)> = Vec::new();
        for c in mesh.children().filter(|c| c.is_element()) {
            let kind = c.tag_name().name();
            if !matches!(kind, "polygons" | "lines" | "linestrips" | "polylist" | "triangles") {
                continue;
            }
            let prim = Primitive::parse(c);
            match by_type.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, list)) => list.push(prim),
                None => by_type.push((kind, vec![prim])),
            }
        }
        let skin = node
            .attribute("id")
            .and_then(|id| self.lib.geometry_skins.get(id).copied())
            .map(|s| self.skin(s));
        let skin_sources = skin.as_ref().and_then(|s| Some((s.vertex_indices.as_ref()?, s.vertex_weights.as_ref()?)));

        let mut parts = Vec::new();
        for (kind, prims) in &by_type {
            let mut source = |id: &str| sources.get(id).copied().and_then(|n| self.source(n));
            let with_uv = prims.iter().filter(|p| p.has_uv).count();
            let uvs_need_fix = with_uv > 0 && with_uv < prims.len();
            let mut positions = Vec::new();
            let mut normals = Vec::new();
            let mut uv0 = Vec::new();
            let mut uv1 = Vec::new();
            let mut skin_index: Vec<f32> = Vec::new();
            let mut skin_weight: Vec<f32> = Vec::new();
            let mut groups = Vec::new();
            let mut material_keys = Vec::new();
            let mut start = 0.0;
            for prim in prims {
                let count = prim.declared_count(kind);
                groups.push((start, count));
                start += count;
                if let Some(m) = prim.material.filter(|m| !m.is_empty()) {
                    material_keys.push(m.to_string());
                }
                // Lenient: inputs naming a missing source are skipped.
                for (name, id, offset) in &prim.inputs {
                    match name.as_str() {
                        "VERTEX" => {
                            for (semantic, vid) in &vertices {
                                let Some(src) = source(vid) else { continue };
                                let data = prim.gather(&src, *offset);
                                match *semantic {
                                    "POSITION" => {
                                        let before = positions.len();
                                        items3(&data, src.stride, &mut positions);
                                        if let Some((si, sw)) = skin_sources {
                                            skin_index.extend(prim.gather(si, *offset));
                                            skin_weight.extend(prim.gather(sw, *offset));
                                        }
                                        // #3803: primitives without UVs next to
                                        // ones with them get (0, 0).
                                        if !prim.has_uv && uvs_need_fix {
                                            uv0.resize(uv0.len() + positions.len() - before, flip_v([0.0, 0.0]));
                                        }
                                    }
                                    "NORMAL" => items3(&data, src.stride, &mut normals),
                                    "TEXCOORD" => items2(&data, src.stride, &mut uv0),
                                    "TEXCOORD1" => items2(&data, src.stride, &mut uv1),
                                    _ => {}
                                }
                            }
                        }
                        "NORMAL" | "TEXCOORD" | "TEXCOORD1" => {
                            let Some(src) = source(id) else { continue };
                            let data = prim.gather(&src, *offset);
                            match name.as_str() {
                                "NORMAL" => items3(&data, src.stride, &mut normals),
                                "TEXCOORD" => items2(&data, src.stride, &mut uv0),
                                _ => items2(&data, src.stride, &mut uv1),
                            }
                        }
                        _ => {}
                    }
                }
            }
            let n = positions.len();
            let skin = (!skin_index.is_empty()).then(|| {
                skin_index.resize(n * 4, 0.0);
                skin_weight.resize(n * 4, 0.0);
                skin_index
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(skin_weight.as_chunks::<4>().0)
                    .map(|(i, w)| {
                        let indices = [i[0] as u32, i[1] as u32, i[2] as u32, i[3] as u32];
                        // normalizeSkinWeights
                        let scale = 1.0 / (w[0].abs() + w[1].abs() + w[2].abs() + w[3].abs());
                        if scale.is_finite() {
                            (indices, [w[0] * scale, w[1] * scale, w[2] * scale, w[3] * scale])
                        } else {
                            (indices, [1.0, 0.0, 0.0, 0.0])
                        }
                    })
                    .collect()
            });
            let kind = match *kind {
                "lines" => PartKind::Lines,
                "linestrips" => PartKind::LineStrip,
                _ => PartKind::Triangles,
            };
            parts.push(Rc::new(Part {
                kind,
                normals: fit(normals, n, [0.0; 3]),
                uv0: fit(uv0, n, [0.0; 2]),
                uv1: fit(uv1, n, [0.0; 2]),
                positions,
                skin,
                groups,
                material_keys,
            }));
        }
        parts
    }

    // -----------------------------------------------------------------------
    // Materials

    /// resolveMaterialBinding + buildObjects' material handling.
    fn resolve_materials(&mut self, part: &Part, bindings: &HashMap<&str, &str>) -> Vec<usize> {
        let mut materials: Vec<usize> = part
            .material_keys
            .iter()
            .map(|key| match bindings.get(key.as_str()) {
                Some(target) => self.material(target),
                None => self.fallback_material(),
            })
            .collect();
        let lines = part.kind != PartKind::Triangles;
        if materials.is_empty() {
            materials.push(if lines { self.default_line_material() } else { self.default_mesh_material() });
        }
        if lines {
            // Phong and Lambert become LineBasicMaterial with the colour,
            // opacity and transparency; other materials stay as they are.
            for m in materials.iter_mut() {
                *m = self.line_material(*m);
            }
        }
        materials
    }

    fn line_material(&mut self, source: usize) -> usize {
        let src = &self.scene.materials[source];
        if !matches!(src.shading, Shading::Phong | Shading::Lambert) {
            return source;
        }
        if let Some(&m) = self.line_material_cache.get(&source) {
            return m;
        }
        let line = Material { color: src.color, opacity: src.opacity, alpha_mode: src.alpha_mode, ..Material::basic() };
        let id = self.scene.add_material(line);
        self.line_material_cache.insert(source, id);
        id
    }

    /// _fallbackMaterial: magenta MeshBasicMaterial for unbound symbols.
    fn fallback_material(&mut self) -> usize {
        if let Some(m) = self.fallback_material {
            return m;
        }
        // 0xff00ff, sRGB hex -> linear stays (1, 0, 1).
        let m = self.scene.add_material(Material { color: srgb3([1.0, 0.0, 1.0]), ..Material::basic() });
        self.fallback_material = Some(m);
        m
    }

    /// `new MeshPhongMaterial()` for geometry without material symbols.
    fn default_mesh_material(&mut self) -> usize {
        if let Some(m) = self.default_mesh_material {
            return m;
        }
        let m = self.scene.add_material(Material::phong());
        self.default_mesh_material = Some(m);
        m
    }

    /// `new LineBasicMaterial()`.
    fn default_line_material(&mut self) -> usize {
        if let Some(m) = self.default_line_material {
            return m;
        }
        let m = self.scene.add_material(Material::basic());
        self.default_line_material = Some(m);
        m
    }

    /// getMaterial: one shared material per <material>. Lenient: an unknown
    /// target (three throws) gets the fallback material.
    fn material(&mut self, id: &str) -> usize {
        let Some(node) = self.lib.materials.get(id).copied() else {
            return self.fallback_material();
        };
        if let Some(&m) = self.material_cache.get(&node.id()) {
            return m;
        }
        let material = self.build_material(node);
        let m = self.scene.add_material(material);
        self.material_cache.insert(node.id(), m);
        m
    }

    /// ColladaComposer.buildMaterial.
    fn build_material(&mut self, node: Node<'a, 'i>) -> Material {
        let effect = last_child(node, "instance_effect")
            .and_then(|e| e.attribute("url"))
            .and_then(|u| self.lib.effects.get(parse_id(u)).copied());
        let profile = effect.and_then(|e| last_child(e, "profile_COMMON"));
        // Lenient: an effect without profile_COMMON / technique throws in
        // three; it gets the default (MeshBasicMaterial) branch here.
        let technique = profile.and_then(|p| last_child(p, "technique"));
        let shader = technique.and_then(|t| {
            t.children()
                .rfind(|c| c.is_element() && matches!(c.tag_name().name(), "constant" | "lambert" | "blinn" | "phong"))
        });
        let shader_name = shader.map(|s| s.tag_name().name());
        let mut m = match shader_name {
            Some("phong" | "blinn") => Material::phong(),
            Some("lambert") => Material::lambert(),
            _ => Material::basic(),
        };
        let phong = m.shading == Shading::Phong;
        let has_emissive = matches!(m.shading, Shading::Phong | Shading::Lambert);

        let mut samplers: HashMap<&str, Node> = HashMap::new();
        let mut surfaces: HashMap<&str, Node> = HashMap::new();
        if let Some(profile) = profile {
            for np in children(profile, "newparam") {
                let sid = np.attribute("sid").unwrap_or("");
                for c in np.children().filter(|c| c.is_element()) {
                    match c.tag_name().name() {
                        "surface" => {
                            surfaces.insert(sid, c);
                        }
                        "sampler2D" => {
                            samplers.insert(sid, c);
                        }
                        _ => {}
                    }
                }
            }
        }
        let ctx = TextureContext { samplers, surfaces };

        let mut params: HashMap<&str, Param> = HashMap::new();
        let mut transparent: Option<(&str, Param)> = None;
        if let Some(shader) = shader {
            for c in shader.children().filter(|c| c.is_element()) {
                match c.tag_name().name() {
                    key @ ("emission" | "diffuse" | "specular" | "bump" | "ambient" | "shininess" | "transparency") => {
                        params.insert(key, Param::parse(c));
                    }
                    "transparent" => transparent = Some((c.attribute("opaque").unwrap_or("A_ONE"), Param::parse(c))),
                    _ => {}
                }
            }
        }

        // Colours are set from the file as is and then converted from sRGB
        // as a whole - which converts three's already linear defaults a
        // second time (the 0x111111 specular ends up at 0.000434).
        let mut color = m.color.to_array();
        let mut specular = m.specular.to_array();
        let mut emissive = m.emissive.to_array();
        for (key, p) in &params {
            match *key {
                "diffuse" => {
                    if let Some(c) = p.rgb() {
                        color = c;
                    }
                    if let Some(t) = p.texture {
                        m.map = self.texture(&ctx, t);
                    }
                }
                "specular" => {
                    if let Some(c) = p.rgb().filter(|_| phong) {
                        specular = c;
                    }
                    if let Some(t) = p.texture {
                        m.specular_map = self.texture(&ctx, t);
                    }
                }
                "bump" => {
                    if let Some(t) = p.texture {
                        m.normal_map = self.texture(&ctx, t);
                    }
                }
                "shininess" => {
                    // `if (parameter.float && material.shininess)`
                    if let Some(f) = p.float.filter(|f| phong && *f != 0.0 && !f.is_nan()) {
                        m.shininess = f as f32;
                    }
                }
                "emission" => {
                    if let Some(c) = p.rgb().filter(|_| has_emissive) {
                        emissive = c;
                    }
                    if let Some(t) = p.texture {
                        m.emissive_map = self.texture(&ctx, t);
                    }
                }
                // "ambient" textures become a lightMap, which the renderer
                // does not support.
                _ => {}
            }
        }
        m.color = srgb3(color);
        if phong {
            m.specular = srgb3(specular);
        }
        if has_emissive {
            m.emissive = srgb3(emissive);
        }

        // <transparent> / <transparency>, each defaulting the other.
        let transparency = params.get("transparency");
        let transparency_value = match (transparency, &transparent) {
            (Some(t), _) => t.float.unwrap_or(f64::NAN),
            (None, Some(_)) => 1.0,
            (None, None) => f64::NAN,
        };
        let default_transparent = Param { color: Some(vec![1.0, 1.0, 1.0, 1.0]), float: None, texture: None };
        let transparent = match (&transparent, transparency) {
            (Some((mode, p)), _) => Some((*mode, p)),
            (None, Some(_)) => Some(("A_ONE", &default_transparent)),
            (None, None) => None,
        };
        if let Some((mode, p)) = transparent {
            if p.texture.is_some() {
                // No alpha map (three #13792), just blending.
                m.alpha_mode = AlphaMode::Blend;
            } else if let Some(c) = &p.color {
                // Lenient: a colour-less <transparent> throws in three.
                let at = |i: usize| c.get(i).copied().unwrap_or(f64::NAN);
                let f = transparency_value;
                let opacity = match mode {
                    "A_ONE" => at(3) * f,
                    "RGB_ZERO" => 1.0 - at(0) * f,
                    "A_ZERO" => 1.0 - at(3) * f,
                    "RGB_ONE" => at(0) * f,
                    _ => 1.0,
                };
                // A NaN opacity leaves three's material opaque.
                if !opacity.is_nan() {
                    m.set_opacity((opacity as f32).min(1.0));
                }
            }
        }

        // <technique><extra><technique>: double_sided and bump.
        let extra = technique.and_then(|t| last_child(t, "extra")).and_then(|e| last_child(e, "technique"));
        if let Some(extra) = extra {
            let mut double_sided = None;
            let mut bump = None;
            for c in extra.children().filter(|c| c.is_element()) {
                match c.tag_name().name() {
                    "double_sided" => double_sided = Some(js_int(&text(c)) == Some(1)),
                    "bump" => bump = Some(last_child(c, "texture")),
                    _ => {}
                }
            }
            if let Some(d) = double_sided {
                m.side = if d { Side::Double } else { Side::Front };
            }
            // Lenient: a bump without <texture> throws in three.
            if let Some(Some(t)) = bump {
                m.normal_map = self.texture(&ctx, t);
                m.normal_scale = Vec2::ONE;
            }
        }
        m
    }

    /// buildMaterial's getTexture: sampler -> surface -> image, or the image
    /// id directly when there is no such sampler.
    fn texture(&mut self, ctx: &TextureContext<'a, 'i>, tex: Node<'a, 'i>) -> Option<TextureRef> {
        let id = tex.attribute("texture").unwrap_or("");
        let image_id: Cow<str> = match ctx.samplers.get(id) {
            Some(sampler) => match last_child(*sampler, "source") {
                // Lenient (three throws without the surface): ids are
                // trimmed.
                Some(source) => {
                    let surface = ctx.surfaces.get(text(source).trim())?;
                    Cow::Owned(text(last_child(*surface, "init_from")?).trim().to_string())
                }
                // Lenient: COLLADA 1.5's <instance_image url="#id">, which
                // three does not read.
                None => Cow::Borrowed(parse_id(child(*sampler, "instance_image")?.attribute("url")?)),
            },
            None => Cow::Borrowed(id),
        };
        let image = self.lib.images.get(image_id.as_ref())?;
        // parseImage: textContent of the first <init_from> (COLLADA 1.5
        // nests the path in <ref>).
        let path = text(child(*image, "init_from")?);
        let path = path.trim();
        let path = path.strip_prefix("file://").map_or(path, |p| {
            // file:///C:/x -> C:/x
            if p.len() > 2 && p.starts_with('/') && p.as_bytes()[2] == b':' { &p[1..] } else { p }
        });
        let index = self.textures.load(&mut self.scene, path, true)?;

        // <texture><extra><technique>: wrap, offset and repeat (Maya).
        let mut settings: HashMap<&str, f64> = HashMap::new();
        for extra in children(tex, "extra") {
            for technique in children(extra, "technique") {
                for c in technique.children().filter(|c| c.is_element()) {
                    let name = c.tag_name().name();
                    let value = match name {
                        "repeatU" | "repeatV" | "offsetU" | "offsetV" => js_float(&text(c)),
                        "wrapU" | "wrapV" => match text(c).to_uppercase().as_str() {
                            "TRUE" => 1.0,
                            "FALSE" => 0.0,
                            t => js_int(t).map_or(f64::NAN, |v| v as f64),
                        },
                        _ => continue,
                    };
                    settings.insert(name, value);
                }
            }
        }
        let mut texture = TextureRef::new(index);
        if !settings.is_empty() {
            // JavaScript truthiness: 0 and NaN are false.
            let get = |k: &str, default: f64| settings.get(k).copied().filter(|v| *v != 0.0 && !v.is_nan()).unwrap_or(default);
            let wrap = |k: &str| if get(k, 0.0) != 0.0 { Wrap::Repeat } else { Wrap::Clamp };
            texture.wrap_u = wrap("wrapU");
            texture.wrap_v = wrap("wrapV");
            let offset = Vec2::new(get("offsetU", 0.0) as f32, get("offsetV", 0.0) as f32);
            let repeat = Vec2::new(get("repeatU", 1.0) as f32, get("repeatV", 1.0) as f32);
            if offset != Vec2::ZERO || repeat != Vec2::ONE {
                texture.transform = UvTransform::three(offset, repeat, 0.0, Vec2::ZERO, true);
            }
        }
        Some(texture)
    }

    // -----------------------------------------------------------------------
    // Output

    /// Computes world matrices like updateWorldMatrix(true, true) from the
    /// loader's root and turns every drawable into meshes.
    fn emit(&mut self, root: usize) -> Result<(), LoadError> {
        // Objects outside the scene graph (bones of a skeleton that is never
        // added) keep the identity matrixWorld three gives them.
        let mut world: Vec<Option<DMat4>> = vec![None; self.objs.len()];
        let mut order = Vec::new();
        let mut stack = vec![(root, DMat4::IDENTITY)];
        while let Some((o, parent)) = stack.pop() {
            if world[o].is_some() {
                continue;
            }
            let w = parent * self.objs[o].local;
            world[o] = Some(w);
            order.push(o);
            for &c in self.objs[o].children.iter().rev() {
                stack.push((c, w));
            }
        }
        let mut vertices = 0usize;
        for o in order {
            let Kind::Drawable(d) = &self.objs[o].kind else { continue };
            let d = d.clone();
            let w = world[o].unwrap_or(DMat4::IDENTITY);
            vertices += d.part.positions.len();
            if vertices > MAX_VERTICES {
                return Err(invalid("dae: too many vertices"));
            }
            if let Some(mesh) = self.mesh(&d, &w, &world) {
                self.scene.meshes.push(mesh);
            }
        }
        Ok(())
    }

    fn mesh(&self, d: &Drawable, world: &DMat4, worlds: &[Option<DMat4>]) -> Option<Mesh> {
        let part = &d.part;
        let n = part.positions.len();
        if n == 0 {
            return None;
        }
        // What WebGLRenderer draws: the whole geometry for a single material,
        // otherwise each group clipped to the vertex count, skipping groups
        // without a material.
        let ranges: Vec<(usize, usize, usize)> = if d.materials.len() == 1 {
            vec![(0, n, d.materials[0])]
        } else {
            part.groups
                .iter()
                .enumerate()
                .filter_map(|(p, &(start, count))| {
                    let material = *d.materials.get(p)?;
                    let s = start.max(0.0);
                    let e = (start + count).min(n as f64);
                    let len = e - s;
                    (len >= 0.0).then_some((s as usize, len as usize, material))
                })
                .collect()
        };

        let skinned = d.skin.as_ref().zip(part.skin.as_ref());
        let (positions, normals) = match skinned {
            Some((binding, skin)) => skin_vertices(binding, skin, part, world, worlds),
            None => (part.positions.clone(), part.normals.clone()),
        };

        let mut mesh = match part.kind {
            PartKind::Triangles | PartKind::Lines => {
                let (topology, unit) = if part.kind == PartKind::Triangles { (Topology::Triangles, 3) } else { (Topology::Lines, 2) };
                let mut mesh = Mesh::new(topology, positions);
                if topology == Topology::Triangles {
                    mesh.normals = normals;
                }
                mesh.uv0 = part.uv0.clone();
                mesh.uv1 = part.uv1.clone();
                mesh.groups = ranges
                    .iter()
                    .map(|&(s, c, material)| Group { start: s as u32, count: (c - c % unit) as u32, material })
                    .collect();
                mesh
            }
            PartKind::LineStrip => strip_segments(&positions, part, &ranges),
        };
        mesh.groups.retain(|g| g.count > 0);
        if mesh.groups.is_empty() {
            return None;
        }

        let world32 = world.as_mat4();
        match skinned {
            None => Some(Placement::new(world32).apply(mesh)),
            Some((binding, _)) => {
                // skin_vertices left the vertices in the mesh's local space;
                // place them, then take the box three gets: computeBoundingBox
                // runs before the first render, while bindMatrixInverse still
                // is inverse(bindShapeMatrix) rather than the mesh's inverse
                // world matrix.
                // (Matrix4.invert gives zeros for a singular matrix.)
                let bind_inverse = if binding.bind_shape.determinant() == 0.0 { DMat4::ZERO } else { binding.bind_shape.inverse() };
                let into_box = bind_inverse * *world;
                let mut local_box = Aabb::EMPTY;
                for p in &mesh.positions {
                    local_box.expand(into_box.transform_point3(DVec3::from(p.map(f64::from))).as_vec3());
                }
                let mut placed = Placement::new(world32).apply(mesh);
                placed.bounds = local_box.transformed(&world32);
                Some(placed)
            }
        }
    }
}

/// Line: a strip through each draw range, as independent segments.
fn strip_segments(positions: &[[f32; 3]], part: &Part, ranges: &[(usize, usize, usize)]) -> Mesh {
    let mut out_pos = Vec::new();
    let mut out_uv0 = part.uv0.as_ref().map(|_| Vec::new());
    let mut out_uv1 = part.uv1.as_ref().map(|_| Vec::new());
    let mut groups = Vec::new();
    for &(start, count, material) in ranges {
        let first = out_pos.len() as u32;
        for i in start..(start + count).saturating_sub(1) {
            for j in [i, i + 1] {
                out_pos.push(positions[j]);
                if let (Some(out), Some(src)) = (out_uv0.as_mut(), part.uv0.as_ref()) {
                    out.push(src[j]);
                }
                if let (Some(out), Some(src)) = (out_uv1.as_mut(), part.uv1.as_ref()) {
                    out.push(src[j]);
                }
            }
        }
        groups.push(Group { start: first, count: out_pos.len() as u32 - first, material });
    }
    let mut mesh = Mesh::new(Topology::Lines, out_pos);
    mesh.uv0 = out_uv0;
    mesh.uv1 = out_uv1;
    mesh.groups = groups;
    mesh
}

/// The rest pose three renders: each vertex is
/// sum(weight * boneWorld * boneInverse) * bindShapeMatrix * position
/// in world space (the SkinnedMesh's own matrix cancels against the
/// bindMatrixInverse it gets on every render). Returned in the mesh's local
/// space for `Placement`; normals follow skinnormal_vertex.
fn skin_vertices(
    binding: &SkinBinding,
    skin: &[SkinVertex],
    part: &Part,
    world: &DMat4,
    worlds: &[Option<DMat4>],
) -> (Vec<[f32; 3]>, Option<Vec<[f32; 3]>>) {
    let bone_matrices: Vec<DMat4> = binding
        .bones
        .iter()
        .map(|b| match b {
            Some((obj, inverse)) => worlds.get(*obj).copied().flatten().unwrap_or(DMat4::IDENTITY) * *inverse,
            None => DMat4::IDENTITY,
        })
        .collect();
    let inverse_world = if world.determinant().abs() > 1e-300 { world.inverse() } else { DMat4::IDENTITY };
    let mut positions = Vec::with_capacity(part.positions.len());
    let mut normals = part.normals.as_ref().map(|n| Vec::with_capacity(n.len()));
    for (i, p) in part.positions.iter().enumerate() {
        let (joints, weights) = skin.get(i).copied().unwrap_or(([0; 4], [0.0; 4]));
        // Bones past the skeleton read zeros from the bone texture.
        let mut skin_matrix = DMat4::ZERO;
        for k in 0..4 {
            if weights[k] != 0.0 {
                let bone = bone_matrices.get(joints[k] as usize).copied().unwrap_or(DMat4::ZERO);
                skin_matrix += bone * weights[k] as f64;
            }
        }
        let to_local = inverse_world * skin_matrix * binding.bind_shape;
        positions.push(to_local.transform_point3(DVec3::from(p.map(f64::from))).as_vec3().to_array());
        if let (Some(out), Some(src)) = (normals.as_mut(), part.normals.as_ref()) {
            out.push(to_local.transform_vector3(DVec3::from(src[i].map(f64::from))).as_vec3().to_array());
        }
    }
    (positions, normals)
}

/// Parsed effect parameter (parseEffectParameter).
struct Param<'a, 'i> {
    color: Option<Vec<f64>>,
    float: Option<f64>,
    texture: Option<Node<'a, 'i>>,
}

impl<'a, 'i> Param<'a, 'i> {
    fn parse(node: Node<'a, 'i>) -> Param<'a, 'i> {
        let mut p = Param { color: None, float: None, texture: None };
        for c in node.children().filter(|c| c.is_element()) {
            match c.tag_name().name() {
                "color" => p.color = Some(floats_f64(&text(c))),
                "float" => p.float = Some(js_float(&text(c))),
                "texture" => p.texture = Some(c),
                _ => {}
            }
        }
        p
    }

    /// Color.fromArray; lenient: fewer than three values (NaN in three) are
    /// ignored.
    fn rgb(&self) -> Option<[f32; 3]> {
        let c = self.color.as_ref()?;
        Some([*c.first()? as f32, *c.get(1)? as f32, *c.get(2)? as f32])
    }
}

struct TextureContext<'a, 'i> {
    samplers: HashMap<&'a str, Node<'a, 'i>>,
    surfaces: HashMap<&'a str, Node<'a, 'i>>,
}

/// <bind_material>: symbol -> material id, from every <instance_material>
/// below it.
fn material_bindings<'a>(inst: Node<'a, '_>) -> HashMap<&'a str, &'a str> {
    let mut map = HashMap::new();
    for bind in children(inst, "bind_material") {
        for im in bind.descendants().filter(|d| d.is_element() && d.tag_name().name() == "instance_material") {
            if let (Some(symbol), Some(target)) = (im.attribute("symbol"), im.attribute("target")) {
                map.insert(symbol, parse_id(target));
            }
        }
    }
    map
}

/// The transform elements of a node, in document order.
fn transforms<'a, 'i>(node: Node<'a, 'i>) -> impl Iterator<Item = (Node<'a, 'i>, &'a str, Vec<f64>)> {
    node.children().filter(|c| c.is_element()).filter_map(|c| {
        let kind = c.tag_name().name();
        matches!(kind, "matrix" | "translate" | "rotate" | "scale").then(|| (c, kind, floats_f64(&text(c))))
    })
}

/// parseNode's data.matrix. Lenient: transforms with too few numbers (NaN
/// in three) are skipped.
fn node_matrix(node: Node) -> DMat4 {
    let mut m = DMat4::IDENTITY;
    for (_, kind, v) in transforms(node) {
        let t = match kind {
            "matrix" => row_major(&v, 0),
            "translate" => vec3_of(&v).map(DMat4::from_translation),
            "rotate" => vec3_of(&v).zip(v.get(3)).map(|(axis, deg)| rotation_axis(axis, deg.to_radians())),
            _ => vec3_of(&v).map(DMat4::from_scale),
        };
        if let Some(t) = t {
            m *= t;
        }
    }
    m
}

const PIVOT_SIDS: [&str; 6] =
    ["rotatePivot", "rotatePivotInverse", "rotatePivotTranslation", "scalePivot", "scalePivotInverse", "scalePivotTranslation"];

fn has_pivot_transforms(node: Node) -> bool {
    transforms(node).any(|(c, _, _)| c.attribute("sid").is_some_and(|s| PIVOT_SIDS.contains(&s)))
}

/// wrapWithTransformHierarchy's chain of Groups, multiplied out. Each sid
/// keeps its last transform (transformData is keyed by sid), every
/// occurrence stays in the order.
fn pivot_chain(node: Node) -> DMat4 {
    let list: Vec<_> = transforms(node).map(|(c, kind, v)| (c.attribute("sid"), kind, v)).collect();
    let mut m = DMat4::IDENTITY;
    for (sid, _, _) in &list {
        let Some((_, kind, v)) = list.iter().rev().find(|(s, _, _)| s == sid) else { continue };
        let t = match *kind {
            "translate" => vec3_of(v).map(DMat4::from_translation),
            "rotate" => vec3_of(v)
                .zip(v.get(3))
                .map(|(axis, deg)| compose(DVec3::ZERO, quat_axis_angle(axis, deg.to_radians()), DVec3::ONE)),
            "scale" => vec3_of(v).map(DMat4::from_scale),
            _ => row_major(v, 0).map(|m| recompose(&m)),
        };
        if let Some(t) = t {
            m *= t;
        }
    }
    m
}

// ---------------------------------------------------------------------------

/// mapbox earcut 3.0.2 as three r186 ships it, for a single contour without
/// holes. The z-order hashing three enables above 80 points only speeds up
/// the ear test; the result is the same.
mod earcut {
    use glam::DVec2;

    struct N {
        i: usize,
        x: f64,
        y: f64,
        prev: usize,
        next: usize,
    }

    struct E {
        n: Vec<N>,
        tris: Vec<usize>,
    }

    pub fn triangulate(points: &[DVec2]) -> Vec<usize> {
        let mut e = E { n: Vec::with_capacity(points.len() + 8), tris: Vec::new() };
        let Some(outer) = e.linked_list(points) else { return Vec::new() };
        if e.n[outer].next == e.n[outer].prev {
            return Vec::new();
        }
        e.earcut_linked(Some(outer), 0);
        e.tris
    }

    fn truthy(v: f64) -> bool {
        v != 0.0 && !v.is_nan()
    }

    impl E {
        fn x(&self, a: usize) -> f64 {
            self.n[a].x
        }
        fn y(&self, a: usize) -> f64 {
            self.n[a].y
        }
        fn next(&self, a: usize) -> usize {
            self.n[a].next
        }
        fn prev(&self, a: usize) -> usize {
            self.n[a].prev
        }

        fn linked_list(&mut self, pts: &[DVec2]) -> Option<usize> {
            // signedArea > 0 for the requested (clockwise) order.
            let mut sum = 0.0;
            let mut j = pts.len().checked_sub(1)?;
            for (i, p) in pts.iter().enumerate() {
                sum += (pts[j].x - p.x) * (p.y + pts[j].y);
                j = i;
            }
            let mut last = None;
            if sum > 0.0 {
                for (i, p) in pts.iter().enumerate() {
                    last = Some(self.insert(i, p.x, p.y, last));
                }
            } else {
                for (i, p) in pts.iter().enumerate().rev() {
                    last = Some(self.insert(i, p.x, p.y, last));
                }
            }
            let mut last = last?;
            if self.equals(last, self.next(last)) {
                self.remove(last);
                last = self.next(last);
            }
            Some(last)
        }

        fn insert(&mut self, i: usize, x: f64, y: f64, last: Option<usize>) -> usize {
            let p = self.n.len();
            match last {
                None => self.n.push(N { i, x, y, prev: p, next: p }),
                Some(l) => {
                    let ln = self.next(l);
                    self.n.push(N { i, x, y, prev: l, next: ln });
                    self.n[ln].prev = p;
                    self.n[l].next = p;
                }
            }
            p
        }

        fn remove(&mut self, p: usize) {
            let (pr, nx) = (self.prev(p), self.next(p));
            self.n[nx].prev = pr;
            self.n[pr].next = nx;
        }

        fn equals(&self, a: usize, b: usize) -> bool {
            self.x(a) == self.x(b) && self.y(a) == self.y(b)
        }

        fn area(&self, p: usize, q: usize, r: usize) -> f64 {
            (self.y(q) - self.y(p)) * (self.x(r) - self.x(q)) - (self.x(q) - self.x(p)) * (self.y(r) - self.y(q))
        }

        fn filter_points(&mut self, start: Option<usize>, end: Option<usize>) -> Option<usize> {
            let start = start?;
            let mut end = end.unwrap_or(start);
            let mut p = start;
            loop {
                let mut again = false;
                if self.equals(p, self.next(p)) || self.area(self.prev(p), p, self.next(p)) == 0.0 {
                    self.remove(p);
                    p = self.prev(p);
                    end = p;
                    if p == self.next(p) {
                        break;
                    }
                    again = true;
                } else {
                    p = self.next(p);
                }
                if !(again || p != end) {
                    break;
                }
            }
            Some(end)
        }

        fn earcut_linked(&mut self, ear: Option<usize>, pass: u8) {
            let Some(mut ear) = ear else { return };
            let mut stop = ear;
            while self.prev(ear) != self.next(ear) {
                let prev = self.prev(ear);
                let next = self.next(ear);
                if self.is_ear(ear) {
                    let (a, b, c) = (self.n[prev].i, self.n[ear].i, self.n[next].i);
                    self.tris.extend([a, b, c]);
                    self.remove(ear);
                    ear = self.next(next);
                    stop = ear;
                    continue;
                }
                ear = next;
                if ear == stop {
                    match pass {
                        0 => {
                            let f = self.filter_points(Some(ear), None);
                            self.earcut_linked(f, 1);
                        }
                        1 => {
                            let f = self.filter_points(Some(ear), None);
                            let cured = self.cure_local_intersections(f);
                            self.earcut_linked(cured, 2);
                        }
                        _ => self.split_earcut(ear),
                    }
                    break;
                }
            }
        }

        fn is_ear(&self, ear: usize) -> bool {
            let (a, b, c) = (self.prev(ear), ear, self.next(ear));
            if self.area(a, b, c) >= 0.0 {
                return false;
            }
            let (ax, bx, cx, ay, by, cy) = (self.x(a), self.x(b), self.x(c), self.y(a), self.y(b), self.y(c));
            let (x0, y0) = (min(min(ax, bx), cx), min(min(ay, by), cy));
            let (x1, y1) = (max(max(ax, bx), cx), max(max(ay, by), cy));
            let mut p = self.next(c);
            while p != a {
                let (px, py) = (self.x(p), self.y(p));
                if px >= x0
                    && px <= x1
                    && py >= y0
                    && py <= y1
                    && point_in_triangle_except_first(ax, ay, bx, by, cx, cy, px, py)
                    && self.area(self.prev(p), p, self.next(p)) >= 0.0
                {
                    return false;
                }
                p = self.next(p);
            }
            true
        }

        fn cure_local_intersections(&mut self, start: Option<usize>) -> Option<usize> {
            let mut start = start?;
            let mut p = start;
            loop {
                let a = self.prev(p);
                let b = self.next(self.next(p));
                if !self.equals(a, b)
                    && self.intersects(a, p, self.next(p), b)
                    && self.locally_inside(a, b)
                    && self.locally_inside(b, a)
                {
                    let (ai, pi, bi) = (self.n[a].i, self.n[p].i, self.n[b].i);
                    self.tris.extend([ai, pi, bi]);
                    let pn = self.next(p);
                    self.remove(p);
                    self.remove(pn);
                    p = b;
                    start = b;
                }
                p = self.next(p);
                if p == start {
                    break;
                }
            }
            self.filter_points(Some(p), None)
        }

        fn split_earcut(&mut self, start: usize) {
            let mut a = start;
            loop {
                let mut b = self.next(self.next(a));
                while b != self.prev(a) {
                    if self.n[a].i != self.n[b].i && self.is_valid_diagonal(a, b) {
                        let c = self.split_polygon(a, b);
                        let an = self.next(a);
                        let a2 = self.filter_points(Some(a), Some(an));
                        let cn = self.next(c);
                        let c2 = self.filter_points(Some(c), Some(cn));
                        self.earcut_linked(a2, 0);
                        self.earcut_linked(c2, 0);
                        return;
                    }
                    b = self.next(b);
                }
                a = self.next(a);
                if a == start {
                    break;
                }
            }
        }

        fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
            let bi = self.n[b].i;
            self.n[self.next(a)].i != bi
                && self.n[self.prev(a)].i != bi
                && !self.intersects_polygon(a, b)
                && ((self.locally_inside(a, b)
                    && self.locally_inside(b, a)
                    && self.middle_inside(a, b)
                    && (truthy(self.area(self.prev(a), a, self.prev(b))) || truthy(self.area(a, self.prev(b), b))))
                    || (self.equals(a, b)
                        && self.area(self.prev(a), a, self.next(a)) > 0.0
                        && self.area(self.prev(b), b, self.next(b)) > 0.0))
        }

        fn intersects(&self, p1: usize, q1: usize, p2: usize, q2: usize) -> bool {
            let o1 = sign(self.area(p1, q1, p2));
            let o2 = sign(self.area(p1, q1, q2));
            let o3 = sign(self.area(p2, q2, p1));
            let o4 = sign(self.area(p2, q2, q1));
            if o1 != o2 && o3 != o4 {
                return true;
            }
            (o1 == 0 && self.on_segment(p1, p2, q1))
                || (o2 == 0 && self.on_segment(p1, q2, q1))
                || (o3 == 0 && self.on_segment(p2, p1, q2))
                || (o4 == 0 && self.on_segment(p2, q1, q2))
        }

        fn on_segment(&self, p: usize, q: usize, r: usize) -> bool {
            self.x(q) <= max(self.x(p), self.x(r))
                && self.x(q) >= min(self.x(p), self.x(r))
                && self.y(q) <= max(self.y(p), self.y(r))
                && self.y(q) >= min(self.y(p), self.y(r))
        }

        fn intersects_polygon(&self, a: usize, b: usize) -> bool {
            let (ai, bi) = (self.n[a].i, self.n[b].i);
            let mut p = a;
            loop {
                let pn = self.next(p);
                let (pi, pni) = (self.n[p].i, self.n[pn].i);
                if pi != ai && pni != ai && pi != bi && pni != bi && self.intersects(p, pn, a, b) {
                    return true;
                }
                p = pn;
                if p == a {
                    return false;
                }
            }
        }

        fn locally_inside(&self, a: usize, b: usize) -> bool {
            if self.area(self.prev(a), a, self.next(a)) < 0.0 {
                self.area(a, b, self.next(a)) >= 0.0 && self.area(a, self.prev(a), b) >= 0.0
            } else {
                self.area(a, b, self.prev(a)) < 0.0 || self.area(a, self.next(a), b) < 0.0
            }
        }

        fn middle_inside(&self, a: usize, b: usize) -> bool {
            let mut p = a;
            let mut inside = false;
            let px = (self.x(a) + self.x(b)) / 2.0;
            let py = (self.y(a) + self.y(b)) / 2.0;
            loop {
                let pn = self.next(p);
                if ((self.y(p) > py) != (self.y(pn) > py))
                    && self.y(pn) != self.y(p)
                    && (px < (self.x(pn) - self.x(p)) * (py - self.y(p)) / (self.y(pn) - self.y(p)) + self.x(p))
                {
                    inside = !inside;
                }
                p = pn;
                if p == a {
                    return inside;
                }
            }
        }

        fn split_polygon(&mut self, a: usize, b: usize) -> usize {
            let a2 = self.n.len();
            self.n.push(N { i: self.n[a].i, x: self.x(a), y: self.y(a), prev: a2, next: a2 });
            let b2 = self.n.len();
            self.n.push(N { i: self.n[b].i, x: self.x(b), y: self.y(b), prev: b2, next: b2 });
            let an = self.next(a);
            let bp = self.prev(b);
            self.n[a].next = b;
            self.n[b].prev = a;
            self.n[a2].next = an;
            self.n[an].prev = a2;
            self.n[b2].next = a2;
            self.n[a2].prev = b2;
            self.n[bp].next = b2;
            self.n[b2].prev = bp;
            b2
        }
    }

    /// Math.min / Math.max, which (unlike f64::min / max) propagate NaN.
    fn min(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    }

    fn max(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    }

    fn sign(v: f64) -> i8 {
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn point_in_triangle_except_first(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, px: f64, py: f64) -> bool {
        !(ax == px && ay == py)
            && (cx - px) * (ay - py) >= (ax - px) * (cy - py)
            && (ax - px) * (by - py) >= (bx - px) * (ay - py)
            && (bx - px) * (cy - py) >= (cx - px) * (by - py)
    }
}
