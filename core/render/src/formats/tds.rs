// Autodesk 3DS: port of three's TDSLoader. Parity details:
//   - colours are taken as they are, no sRGB conversion (the loader calls
//     setRGB without a colour space);
//   - a mesh's material groups are laid out back to back in index order,
//     as the loader assumes, and a group whose material is unknown is
//     simply not drawn;
//   - MESH_MATRIX is applied and undone again, so only its mirroring
//     survives (it flips the winding three culls with);
//   - normals are smoothed over shared vertices (computeVertexNormals on the
//     indexed geometry); the file is not converted from Z-up.

use std::collections::HashMap;
use std::path::Path;

use glam::{Mat4, Vec2, Vec3};

use super::{invalid, LoadError};
use crate::scene::{flip_v, Group, Material, Mesh, Placement, Scene, Side, TextureRef, Topology, UvTransform};
use crate::texture::Textures;

const M3DMAGIC: u16 = 0x4D4D;
const MLIBMAGIC: u16 = 0x3DAA;
const CMAGIC: u16 = 0xC23D;
const COLOR_F: u16 = 0x0010;
const COLOR_24: u16 = 0x0011;
const LIN_COLOR_24: u16 = 0x0012;
const LIN_COLOR_F: u16 = 0x0013;
const INT_PERCENTAGE: u16 = 0x0030;
const FLOAT_PERCENTAGE: u16 = 0x0031;
const MDATA: u16 = 0x3D3D;
const MASTER_SCALE: u16 = 0x0100;
const MAT_ENTRY: u16 = 0xAFFF;
const MAT_NAME: u16 = 0xA000;
const MAT_AMBIENT: u16 = 0xA010;
const MAT_DIFFUSE: u16 = 0xA020;
const MAT_SPECULAR: u16 = 0xA030;
const MAT_SHININESS: u16 = 0xA040;
const MAT_TRANSPARENCY: u16 = 0xA050;
const MAT_TWO_SIDE: u16 = 0xA081;
const MAT_ADDITIVE: u16 = 0xA083;
const MAT_TEXMAP: u16 = 0xA200;
const MAT_SPECMAP: u16 = 0xA204;
const MAT_OPACMAP: u16 = 0xA210;
const MAT_BUMPMAP: u16 = 0xA230;
const MAT_MAPNAME: u16 = 0xA300;
const MAT_MAP_USCALE: u16 = 0xA354;
const MAT_MAP_VSCALE: u16 = 0xA356;
const MAT_MAP_UOFFSET: u16 = 0xA358;
const MAT_MAP_VOFFSET: u16 = 0xA35A;
const NAMED_OBJECT: u16 = 0x4000;
const N_TRI_OBJECT: u16 = 0x4100;
const POINT_ARRAY: u16 = 0x4110;
const FACE_ARRAY: u16 = 0x4120;
const MSH_MAT_GROUP: u16 = 0x4130;
const TEX_VERTS: u16 = 0x4140;
const MESH_MATRIX: u16 = 0x4160;

/// A chunk: id, and a cursor over its body.
struct Chunk<'a> {
    data: &'a [u8],
    id: u16,
    position: usize,
    end: usize,
}

impl<'a> Chunk<'a> {
    fn at(data: &'a [u8], offset: usize) -> Option<Chunk<'a>> {
        let id = u16::from_le_bytes([*data.get(offset)?, *data.get(offset + 1)?]);
        let size = u32::from_le_bytes([*data.get(offset + 2)?, *data.get(offset + 3)?, *data.get(offset + 4)?, *data.get(offset + 5)?]) as usize;
        if size < 6 {
            return None;
        }
        Some(Chunk { data, id, position: offset + 6, end: (offset + size).min(data.len()) })
    }

    fn end_of_chunk(&self) -> bool {
        self.position >= self.end
    }

    fn read_chunk(&mut self) -> Option<Chunk<'a>> {
        if self.end_of_chunk() {
            return None;
        }
        let next = Chunk::at(self.data, self.position)?;
        self.position += next.end - (next.position - 6);
        if next.end > self.end {
            return None;
        }
        Some(next)
    }

    fn byte(&mut self) -> u8 {
        let v = self.data.get(self.position).copied().unwrap_or(0);
        self.position += 1;
        v
    }

    fn word(&mut self) -> u16 {
        let v = match self.data.get(self.position..self.position + 2) {
            Some(b) => u16::from_le_bytes([b[0], b[1]]),
            None => 0,
        };
        self.position += 2;
        v
    }

    fn short(&mut self) -> i16 {
        self.word() as i16
    }

    fn float(&mut self) -> f32 {
        let v = match self.data.get(self.position..self.position + 4) {
            Some(b) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            None => 0.0,
        };
        self.position += 4;
        v
    }

    fn string(&mut self) -> String {
        let mut s = Vec::new();
        loop {
            if self.position >= self.data.len() {
                break;
            }
            let c = self.byte();
            if c == 0 {
                break;
            }
            s.push(c);
        }
        // String.fromCharCode per byte: Latin-1.
        s.into_iter().map(|b| b as char).collect()
    }
}

struct RawMesh {
    positions: Vec<[f32; 3]>,
    uvs: Option<Vec<[f32; 2]>>,
    indices: Vec<u32>,
    /// Material names of the MSH_MAT_GROUP chunks with their face counts.
    groups: Vec<(String, usize)>,
    /// MESH_MATRIX has a negative determinant.
    mirrored: bool,
}

struct Loader<'s> {
    scene: &'s mut Scene,
    textures: &'s mut Textures,
    materials: HashMap<String, usize>,
    meshes: Vec<RawMesh>,
    master_scale: f32,
}

pub fn load(path: &Path, textures: &mut Textures) -> Result<Scene, LoadError> {
    let data = std::fs::read(path)?;
    let mut scene = Scene::default();
    let mut root = Chunk::at(&data, 0).ok_or_else(|| invalid("3DS too short"))?;
    if root.id != MLIBMAGIC && root.id != CMAGIC && root.id != M3DMAGIC {
        return Err(invalid("not a 3DS file"));
    }
    let mut loader = Loader { scene: &mut scene, textures, materials: HashMap::new(), meshes: Vec::new(), master_scale: 1.0 };
    while let Some(next) = root.read_chunk() {
        if next.id == MDATA {
            loader.read_mesh_data(next);
        }
    }
    let Loader { meshes, materials, master_scale, .. } = loader;
    let group_scale = Mat4::from_scale(Vec3::splat(master_scale));
    let default_material = scene.materials.len();
    let mut default_used = false;
    for raw in meshes {
        let mut mesh = Mesh::new(Topology::Triangles, raw.positions).with_indices(raw.indices);
        mesh.uv0 = raw.uvs.filter(|u| u.len() == mesh.positions.len());
        if mesh.indices.as_ref().is_some_and(|i| i.iter().any(|&v| v as usize >= mesh.positions.len())) {
            continue;
        }
        // computeVertexNormals() on the indexed geometry, in the mesh's own
        // space, as the loader does before placing it.
        crate::scene::compute_vertex_normals(&mut mesh);
        if raw.groups.is_empty() {
            default_used = true;
            mesh = mesh.with_material(default_material);
        } else {
            let mut start = 0u32;
            let mut groups = Vec::new();
            for (name, faces) in &raw.groups {
                let count = (*faces * 3) as u32;
                // A group whose material is unknown gets no material: three
                // draws nothing for it.
                if let Some(&m) = materials.get(name) {
                    groups.push(Group { start, count, material: m });
                }
                start += count;
            }
            if groups.is_empty() {
                continue;
            }
            mesh.groups = groups;
        }
        // The vertices stay where the file put them: the loader applies the
        // inverse of MESH_MATRIX to the geometry and the matrix to the mesh.
        // Only a mirroring matrix changes anything - through the winding.
        let mirrored = raw.mirrored;
        let mut placed = Placement::new(group_scale).apply(mesh);
        if mirrored {
            // Same flip three's frontFaceCW applies for the mesh's mirrored
            // matrix world; the geometry's own inverse transform does not
            // flip anything at draw time.
            if let Some(indices) = placed.indices.as_mut() {
                for tri in indices.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
            if let Some(normals) = placed.normals.as_mut() {
                for n in normals.iter_mut() {
                    *n = [-n[0], -n[1], -n[2]];
                }
            }
        }
        scene.meshes.push(placed);
    }
    if default_used {
        scene.materials.push(Material::phong());
    }
    Ok(scene)
}


impl<'s> Loader<'s> {
    fn read_mesh_data(&mut self, mut chunk: Chunk) {
        while let Some(mut next) = chunk.read_chunk() {
            match next.id {
                MASTER_SCALE => self.master_scale = next.float(),
                NAMED_OBJECT => self.read_named_object(next),
                MAT_ENTRY => self.read_material_entry(next),
                _ => {}
            }
        }
    }

    fn read_named_object(&mut self, mut chunk: Chunk) {
        let _name = chunk.string();
        while let Some(next) = chunk.read_chunk() {
            if next.id == N_TRI_OBJECT {
                let mesh = self.read_mesh(next);
                self.meshes.push(mesh);
            }
        }
    }

    fn read_material_entry(&mut self, mut chunk: Chunk) {
        let mut m = Material::phong();
        let mut name = String::new();
        while let Some(mut next) = chunk.read_chunk() {
            match next.id {
                MAT_NAME => name = next.string(),
                MAT_TWO_SIDE => m.side = Side::Double,
                MAT_ADDITIVE => m.additive = true,
                MAT_DIFFUSE | MAT_AMBIENT => m.color = read_color(next),
                MAT_SPECULAR => m.specular = read_color(next),
                MAT_SHININESS => m.shininess = read_percentage(next) * 100.0,
                MAT_TRANSPARENCY => {
                    let t = read_percentage(next);
                    m.opacity = 1.0 - t;
                    m.alpha_mode = if m.opacity < 1.0 { crate::scene::AlphaMode::Blend } else { crate::scene::AlphaMode::Opaque };
                }
                MAT_TEXMAP => m.map = self.read_map(next),
                MAT_BUMPMAP => m.bump_map = self.read_map(next),
                MAT_OPACMAP => m.alpha_map = self.read_map(next),
                MAT_SPECMAP => m.specular_map = self.read_map(next),
                _ => {}
            }
        }
        let id = self.scene.add_material(m);
        self.materials.insert(name, id);
    }

    fn read_map(&mut self, mut chunk: Chunk) -> Option<TextureRef> {
        let mut image = None;
        let (mut offset, mut repeat) = (Vec2::ZERO, Vec2::ONE);
        while let Some(mut next) = chunk.read_chunk() {
            match next.id {
                MAT_MAPNAME => {
                    let name = next.string();
                    image = self.textures.load(self.scene, &name, false);
                }
                MAT_MAP_UOFFSET => offset.x = next.float(),
                MAT_MAP_VOFFSET => offset.y = next.float(),
                MAT_MAP_USCALE => repeat.x = next.float(),
                MAT_MAP_VSCALE => repeat.y = next.float(),
                _ => {}
            }
        }
        let mut t = TextureRef::new(image?);
        if repeat.is_finite() && offset.is_finite() {
            t.transform = UvTransform::three(offset, repeat, 0.0, Vec2::ZERO, true);
        }
        // TextureLoader's default ClampToEdge wrapping.
        t.wrap_u = crate::scene::Wrap::Clamp;
        t.wrap_v = crate::scene::Wrap::Clamp;
        Some(t)
    }

    fn read_mesh(&mut self, mut chunk: Chunk) -> RawMesh {
        let mut mesh = RawMesh { positions: Vec::new(), uvs: None, indices: Vec::new(), groups: Vec::new(), mirrored: false };
        while let Some(mut next) = chunk.read_chunk() {
            match next.id {
                POINT_ARRAY => {
                    let points = next.word() as usize;
                    mesh.positions = (0..points).map(|_| [next.float(), next.float(), next.float()]).collect();
                }
                FACE_ARRAY => self.read_face_array(next, &mut mesh),
                TEX_VERTS => {
                    let texels = next.word() as usize;
                    mesh.uvs = Some((0..texels).map(|_| flip_v([next.float(), next.float()])).collect());
                }
                MESH_MATRIX => {
                    let v: Vec<f32> = (0..12).map(|_| next.float()).collect();
                    // TDSLoader permutes rows and columns alike while
                    // filling its Matrix4, so the determinant is that of the
                    // three axis vectors as stored.
                    let axes = glam::Mat3::from_cols_array(&[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]]);
                    mesh.mirrored = axes.determinant() < 0.0;
                }
                _ => {}
            }
        }
        mesh
    }

    fn read_face_array(&mut self, mut chunk: Chunk, mesh: &mut RawMesh) {
        let faces = chunk.word() as usize;
        for _ in 0..faces {
            let (a, b, c) = (chunk.word(), chunk.word(), chunk.word());
            mesh.indices.extend([a as u32, b as u32, c as u32]);
            chunk.word(); // visibility
        }
        while !chunk.end_of_chunk() {
            let Some(mut sub) = chunk.read_chunk() else { break };
            if sub.id == MSH_MAT_GROUP {
                let name = sub.string();
                let count = sub.word() as usize;
                mesh.groups.push((name, count));
            }
        }
    }
}

fn read_color(mut chunk: Chunk) -> Vec3 {
    let Some(mut sub) = chunk.read_chunk() else { return Vec3::ONE };
    match sub.id {
        COLOR_24 | LIN_COLOR_24 => Vec3::new(sub.byte() as f32 / 255.0, sub.byte() as f32 / 255.0, sub.byte() as f32 / 255.0),
        COLOR_F | LIN_COLOR_F => Vec3::new(sub.float(), sub.float(), sub.float()),
        // An unknown sub-chunk yields new Color(): white.
        _ => Vec3::ONE,
    }
}

fn read_percentage(mut chunk: Chunk) -> f32 {
    let Some(mut sub) = chunk.read_chunk() else { return 0.0 };
    match sub.id {
        INT_PERCENTAGE => sub.short() as f32 / 100.0,
        FLOAT_PERCENTAGE => sub.float(),
        _ => 0.0,
    }
}
