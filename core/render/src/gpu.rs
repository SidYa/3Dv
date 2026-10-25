// Offscreen wgpu renderer for thumbnails: draws a finished Scene the way
// Thumbnail.svelte does with three.js - square frame, 60° camera looking from
// scene.ts's VIEW_DIRECTION, studio lights plus the baked RoomEnvironment,
// 4x MSAA. Render lists as in WebGLRenderer: opaque objects, then
// transmissive ones (glass, refracting a mipmapped render of the opaque
// scene, three's transmission pass), then transparent ones back to front.
//
// One Renderer owns a device and is reused for many thumbnails; pipelines
// and the environment textures are created once.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::scene::{srgb_to_linear, AlphaMode, Framing, Image, Material, Scene, Shading, Side, Topology, Wrap};

/// Thumbnail.svelte / scene.ts constants.
const BACKGROUND: f64 = 0x1a as f64 / 255.0;
const FOV_DEGREES: f32 = 60.0;
const FRAME_PADDING: f32 = 1.08;
const ENVIRONMENT_INTENSITY: f32 = 0.5;
const SAMPLES: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// three's transmission render target is HalfFloat.
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Thumbnail.svelte's empty-frame probe.
const PROBE_TOLERANCE: i32 = 5;
const PROBE_STEP: u32 = 4;
/// Material texture slots, in shader order.
const SLOTS: usize = 10;

#[derive(Debug)]
pub enum RenderError {
    /// No usable GPU or software adapter.
    NoAdapter(String),
    /// The model drew nothing but background.
    EmptyFrame,
    /// Too large for the device, or a device error.
    Gpu(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::NoAdapter(s) => write!(f, "no adapter: {s}"),
            RenderError::EmptyFrame => write!(f, "empty frame"),
            RenderError::Gpu(s) => write!(f, "gpu: {s}"),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv0: [f32; 2],
    uv1: [f32; 2],
    color: [f32; 4],
    tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointVertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    light_dir: [[f32; 4]; 3],
    light_color: [[f32; 4]; 3],
    ambient: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    color: [f32; 4],
    emissive: [f32; 4],
    specular: [f32; 4],
    pbr: [f32; 4],
    extra: [f32; 4],
    flags: [u32; 4],
    transmission: [f32; 4],
    attenuation: [f32; 4],
    uv_rows: [[f32; 4]; SLOTS * 2],
}

const F_VERTEX_COLORS: u32 = 1;
const F_FLAT: u32 = 2;
const F_DOUBLE_SIDED: u32 = 4;
const F_BACK_SIDE: u32 = 8;
const F_OPAQUE: u32 = 16;
const F_SIZE_ATTENUATION: u32 = 32;
const F_ALPHA_TEST: u32 = 64;
const F_CULL_BACK: u32 = 256;
const F_CULL_FRONT: u32 = 512;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Blend {
    None,
    Normal,
    Additive,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    topology: u8,
    /// The pipeline carries the transmission code (USE_TRANSMISSION).
    transmission: bool,
    blend: Blend,
    depth_write: bool,
    /// Drawing the linear transmission source instead of the 8-bit canvas.
    hdr: bool,
}

/// How a material is drawn in one pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Pass {
    /// As the material says.
    Normal,
    /// A transparent double-sided material's back faces (three draws them
    /// as BackSide first), or a transmissive one's for the transmission
    /// source.
    BackFaces,
    /// Then its front faces.
    FrontFaces,
}

#[derive(Clone, Copy)]
struct Draw {
    mesh: usize,
    start: u32,
    count: u32,
    material: usize,
    pass: Pass,
    depth: f32,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    frame_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    mip_pipeline: wgpu::RenderPipeline,
    mip_layout: wgpu::BindGroupLayout,
    env_view: wgpu::TextureView,
    dfg_view: wgpu::TextureView,
    linear_clamp: wgpu::Sampler,
    mip_sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    samplers: HashMap<(Wrap, Wrap), wgpu::Sampler>,
    max_buffer: u64,
    max_texture: u32,
    pub adapter: String,
}

const MIP_SHADER: &str = r#"
// A mip level from the one above it: bilinear at the centre of each 2x2
// block, the box filter of generateMipmap.
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
@fragment fn fs(o: Out) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, smp, o.uv, 0.0);
}
"#;

impl Renderer {
    /// Creates a renderer on the best available GPU, or on a software
    /// adapter (WARP on Windows, llvmpipe/lavapipe elsewhere) when `software`
    /// is set or no GPU is usable.
    pub fn new(software: bool) -> Result<Renderer, RenderError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        // No validation layers in release builds. On DX12 the DEBUG flag only
        // makes FXC skip optimization, which cuts the first thumbnail's
        // shader compile by a third; GPU time is negligible at this size and
        // the output is the same.
        desc.flags = if cfg!(debug_assertions) {
            wgpu::InstanceFlags::debugging()
        } else if cfg!(windows) {
            wgpu::InstanceFlags::DEBUG
        } else {
            wgpu::InstanceFlags::empty()
        };
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: software,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: !software,
                compatible_surface: None,
                apply_limit_buckets: false,
            }))
        })
        .map_err(|e| RenderError::NoAdapter(e.to_string()))?;
        let info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("3Dv thumbnails"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| RenderError::NoAdapter(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("three.js port"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[uniform_entry(0), texture_entry(1), sampler_entry(2), texture_entry(3), texture_entry(4), sampler_entry(5)],
        });
        let mut material_entries = vec![uniform_entry(0)];
        for i in 0..SLOTS as u32 {
            material_entries.push(texture_entry(1 + i));
        }
        for i in 0..SLOTS as u32 {
            material_entries.push(sampler_entry(1 + SLOTS as u32 + i));
        }
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("material"), entries: &material_entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thumbnail"),
            bind_group_layouts: &[Some(&frame_layout), Some(&material_layout)],
            immediate_size: 0,
        });

        let mip_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mipmaps"), source: wgpu::ShaderSource::Wgsl(MIP_SHADER.into()) });
        let mip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mipmaps"), entries: &[texture_entry(0), sampler_entry(1)] });
        let mip_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mipmaps"), bind_group_layouts: &[Some(&mip_layout)], immediate_size: 0 });
        let mip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mipmaps"),
            layout: Some(&mip_pipeline_layout),
            vertex: wgpu::VertexState { module: &mip_module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &mip_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let (env_w, env_h, env_texels) = crate::env::room_pmrem();
        let env_view = create_texture(&device, &queue, "room pmrem", env_w, env_h, HDR_FORMAT, &[(env_w, env_h, bytemuck::cast_slice(&env_texels).to_vec())]);
        let dfg = crate::env::dfg_lut();
        let dfg_view = create_texture(&device, &queue, "dfg lut", 16, 16, wgpu::TextureFormat::Rg16Float, &[(16, 16, bytemuck::cast_slice(&dfg).to_vec())]);
        let white = create_texture(&device, &queue, "white", 1, 1, wgpu::TextureFormat::Rgba8Unorm, &[(1, 1, vec![255, 255, 255, 255])]);
        let linear_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear mip clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Ok(Renderer {
            device,
            queue,
            shader,
            frame_layout,
            material_layout,
            pipeline_layout,
            pipelines: HashMap::new(),
            mip_pipeline,
            mip_layout,
            env_view,
            dfg_view,
            linear_clamp,
            mip_sampler,
            white,
            samplers: HashMap::new(),
            max_buffer: limits.max_buffer_size,
            max_texture: limits.max_texture_dimension_2d,
            adapter: format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type),
        })
    }

    /// Compiles the pipelines nearly every model needs, so the first
    /// thumbnail does not wait for them.
    pub fn warm_up(&mut self) {
        self.pipeline(PipelineKey { topology: 0, transmission: false, blend: Blend::None, depth_write: true, hdr: false });
    }

    fn pipeline(&mut self, key: PipelineKey) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&key) {
            let p = self.create_pipeline(key);
            self.pipelines.insert(key, p);
        }
        &self.pipelines[&key]
    }

    fn create_pipeline(&self, key: PipelineKey) -> wgpu::RenderPipeline {
        let mesh_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2, 4 => Float32x4, 5 => Float32x4];
        let point_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let points = key.topology == 2;
        let buffers = if points {
            [Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<PointVertex>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &point_attrs })]
        } else {
            [Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &mesh_attrs })]
        };
        let blend = match key.blend {
            Blend::None => None,
            // NormalBlending: blendFuncSeparate(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA)
            Blend::Normal => Some(wgpu::BlendState {
                color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            }),
            // AdditiveBlending: blendFunc(SRC_ALPHA, ONE)
            Blend::Additive => Some(wgpu::BlendState {
                color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            }),
        };
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("thumbnail"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some(if points { "vs_points" } else { "vs_main" }),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: match key.topology {
                    1 => wgpu::PrimitiveTopology::LineList,
                    _ => wgpu::PrimitiveTopology::TriangleList,
                },
                front_face: wgpu::FrontFace::Ccw,
                // Culled in the fragment shader, see F_CULL_BACK.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(key.depth_write),
                // three's LessEqualDepth.
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &[("USE_TRANSMISSION", if key.transmission { 1.0 } else { 0.0 })], ..Default::default() },
                targets: &[Some(wgpu::ColorTargetState { format: if key.hdr { HDR_FORMAT } else { COLOR_FORMAT }, blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn sampler(&mut self, wrap_u: Wrap, wrap_v: Wrap) -> wgpu::Sampler {
        let address = |w: Wrap| match w {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::Clamp => wgpu::AddressMode::ClampToEdge,
            Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
        };
        self.samplers
            .entry((wrap_u, wrap_v))
            .or_insert_with(|| {
                self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("texture"),
                    address_mode_u: address(wrap_u),
                    address_mode_v: address(wrap_v),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                })
            })
            .clone()
    }

    fn frame_group(&self, frame: &FrameUniform, transmission: &wgpu::TextureView) -> wgpu::BindGroup {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("frame"), contents: bytemuck::bytes_of(frame), usage: wgpu::BufferUsages::UNIFORM });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &self.frame_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.env_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.linear_clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.dfg_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(transmission) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.mip_sampler) },
            ],
        })
    }

    /// Renders the scene into a `size`×`size` RGBA image (top row first).
    pub fn render(&mut self, scene: &Scene, framing: &Framing, size: u32) -> Result<Vec<u8>, RenderError> {
        let size = size.clamp(16, self.max_texture.min(4096));

        // --- camera: scene.ts frameObject() ---------------------------------
        let view_direction = Vec3::new(0.35, 0.3, 1.0).normalize();
        let fov = FOV_DEGREES.to_radians();
        let distance = (framing.radius / (fov / 2.0).sin()) * FRAME_PADDING;
        let eye = view_direction * distance + framing.center;
        let near = (distance / 1000.0).max(1e-5);
        let far = (distance + framing.radius) * 10.0;
        let view = look_at(eye, framing.center, Vec3::Y);
        let proj = perspective(fov, 1.0, near, far);

        // --- lights: scene.ts addStudioLights() -------------------------------
        let lights = [(Vec3::new(1.0, 2.0, 3.0), 1.5f32), (Vec3::new(-1.0, -0.5, -2.0), 0.25), (Vec3::new(-1.0, 1.0, 1.0), 0.3)];
        let mut light_dir = [[0.0f32; 4]; 3];
        let mut light_color = [[0.0f32; 4]; 3];
        for (i, (pos, intensity)) in lights.iter().enumerate() {
            // DirectionalLight aims at the origin; three passes the
            // direction in view space.
            let d = view.transform_vector3(pos.normalize()).normalize();
            light_dir[i] = [d.x, d.y, d.z, 0.0];
            light_color[i] = [*intensity, *intensity, *intensity, 0.0];
        }
        let frame = FrameUniform {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            light_dir,
            light_color,
            ambient: [0.12, 0.12, 0.12, 0.0],
            params: [ENVIRONMENT_INTENSITY, size as f32, size as f32, 0.0],
            params2: [0.0; 4],
        };

        // --- geometry -----------------------------------------------------------
        struct Gpu {
            vertices: wgpu::Buffer,
            indices: Option<wgpu::Buffer>,
            points: bool,
        }
        let mut meshes = Vec::with_capacity(scene.meshes.len());
        let mut total = 0u64;
        for mesh in &scene.meshes {
            let n = mesh.positions.len();
            let (vertices, points) = if mesh.topology == Topology::Points {
                let data: Vec<PointVertex> = (0..n).map(|i| PointVertex { position: mesh.positions[i], color: mesh.colors.as_ref().map_or([1.0; 4], |c| c[i]) }).collect();
                total += (data.len() * std::mem::size_of::<PointVertex>()) as u64;
                (self.buffer(bytemuck::cast_slice(&data), wgpu::BufferUsages::VERTEX)?, true)
            } else {
                let data: Vec<Vertex> = (0..n)
                    .map(|i| Vertex {
                        position: mesh.positions[i],
                        normal: mesh.normals.as_ref().map_or([0.0, 0.0, 1.0], |v| v[i]),
                        uv0: mesh.uv0.as_ref().map_or([0.0; 2], |v| v[i]),
                        uv1: mesh.uv1.as_ref().map_or([0.0; 2], |v| v[i]),
                        color: mesh.colors.as_ref().map_or([1.0; 4], |v| v[i]),
                        tangent: mesh.tangents.as_ref().map_or([0.0; 4], |v| v[i]),
                    })
                    .collect();
                total += (data.len() * std::mem::size_of::<Vertex>()) as u64;
                (self.buffer(bytemuck::cast_slice(&data), wgpu::BufferUsages::VERTEX)?, false)
            };
            let indices = match (&mesh.indices, points) {
                (Some(i), false) => Some(self.buffer(bytemuck::cast_slice(i), wgpu::BufferUsages::INDEX)?),
                _ => None,
            };
            meshes.push(Gpu { vertices, indices, points });
            if total > 4 * self.max_buffer {
                return Err(RenderError::Gpu("model too large".into()));
            }
        }

        // --- render lists (WebGLRenderLists) ----------------------------------
        let mut opaque = Vec::new();
        let mut transmissive = Vec::new();
        let mut transparent = Vec::new();
        for (mi, mesh) in scene.meshes.iter().enumerate() {
            let z = view.transform_point3(mesh.origin).z;
            for g in &mesh.groups {
                let m = &scene.materials[g.material];
                let d = |pass| Draw { mesh: mi, start: g.start, count: g.count, material: g.material, pass, depth: z };
                if m.is_transmissive() && mesh.topology == Topology::Triangles {
                    transmissive.push(d(Pass::Normal));
                } else if m.is_transparent() {
                    if m.side == Side::Double && mesh.topology == Topology::Triangles {
                        transparent.push(d(Pass::BackFaces));
                        transparent.push(d(Pass::FrontFaces));
                    } else {
                        transparent.push(d(Pass::Normal));
                    }
                } else {
                    opaque.push(d(Pass::Normal));
                }
            }
        }
        // Back to front: the most negative view-space z first. The sort is
        // stable, so the back/front passes of one object stay in order.
        let back_to_front = |a: &Draw, b: &Draw| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal);
        transmissive.sort_by(back_to_front);
        transparent.sort_by(back_to_front);

        let mut cache = Cache::default();
        let buffers: Vec<(&wgpu::Buffer, Option<&wgpu::Buffer>, bool)> = meshes.iter().map(|g| (&g.vertices, g.indices.as_ref(), g.points)).collect();

        let extent = wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thumbnail") });

        // --- three's renderTransmissionPass --------------------------------------
        let transmission_view = if transmissive.is_empty() {
            self.white.clone()
        } else {
            // The opaque scene, linear and not tone mapped, into a mipmapped
            // HalfFloat target; then the back faces of double-sided
            // transmissive objects on top of it (they refract that first
            // render), resolved into a second texture to avoid sampling the
            // one being written.
            let hdr_frame = FrameUniform { params2: [1.0, 0.0, 0.0, 0.0], ..frame };
            let msaa = self.target("transmission msaa", extent, HDR_FORMAT, SAMPLES, 1);
            let depth = self.target("transmission depth", extent, DEPTH_FORMAT, SAMPLES, 1);
            let levels = 32 - size.leading_zeros();
            let first = self.target("transmission", extent, HDR_FORMAT, 1, levels);
            let keys = self.prepare(scene, &opaque, true, &mut cache);
            let white = self.white.clone();
            let group = self.frame_group(&hdr_frame, &white);
            self.draw_pass(&mut encoder, &msaa, &first, &depth, true, &group, &opaque, &keys, &buffers, &cache.groups);
            self.mipmaps(&mut encoder, &first, levels);

            let back: Vec<Draw> = transmissive
                .iter()
                .filter(|d| scene.materials[d.material].side == Side::Double)
                .map(|d| Draw { pass: Pass::BackFaces, ..*d })
                .collect();
            if back.is_empty() {
                first.create_view(&Default::default())
            } else {
                let second = self.target("transmission 2", extent, HDR_FORMAT, 1, levels);
                let keys = self.prepare(scene, &back, true, &mut cache);
                let group = self.frame_group(&hdr_frame, &first.create_view(&Default::default()));
                self.draw_pass(&mut encoder, &msaa, &second, &depth, false, &group, &back, &keys, &buffers, &cache.groups);
                self.mipmaps(&mut encoder, &second, levels);
                second.create_view(&Default::default())
            }
        };

        // --- the canvas ---------------------------------------------------------------
        let all: Vec<Draw> = opaque.iter().chain(transmissive.iter()).chain(transparent.iter()).copied().collect();
        let keys = self.prepare(scene, &all, false, &mut cache);
        let msaa = self.target("msaa", extent, COLOR_FORMAT, SAMPLES, 1);
        let resolve = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resolve"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = self.target("depth", extent, DEPTH_FORMAT, SAMPLES, 1);
        let group = self.frame_group(&frame, &transmission_view);
        self.draw_pass(&mut encoder, &msaa, &resolve, &depth, true, &group, &all, &keys, &buffers, &cache.groups);

        // --- read back ---------------------------------------------------------------
        let row = size * 4;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &resolve, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(size) } },
            extent,
        );
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| RenderError::Gpu(e.to_string()))?;
        rx.recv().map_err(|e| RenderError::Gpu(e.to_string()))?.map_err(|e| RenderError::Gpu(e.to_string()))?;
        let mut pixels = Vec::with_capacity((row * size) as usize);
        {
            let data = slice.get_mapped_range().map_err(|e| RenderError::Gpu(e.to_string()))?;
            for y in 0..size {
                let start = (y * padded) as usize;
                pixels.extend_from_slice(&data[start..start + row as usize]);
            }
        }
        readback.unmap();
        // The canvas is opaque; transparent fragments leave their alpha in
        // the buffer, which a PNG must not carry.
        for px in pixels.chunks_exact_mut(4) {
            px[3] = 255;
        }
        if is_frame_empty(&pixels, size) {
            return Err(RenderError::EmptyFrame);
        }
        Ok(pixels)
    }

    /// Material bind groups and pipelines for a list of draws; returns the
    /// pipeline of each draw.
    fn prepare(&mut self, scene: &Scene, draws: &[Draw], hdr: bool, cache: &mut Cache) -> Vec<PipelineKey> {
        let mut keys = Vec::with_capacity(draws.len());
        for d in draws {
            let topology = scene.meshes[d.mesh].topology;
            if !cache.groups.contains_key(&(d.material, d.pass)) {
                let group = self.material_group(scene, d.material, d.pass, topology, &mut cache.textures);
                cache.groups.insert((d.material, d.pass), group);
            }
            let key = PipelineKey { hdr, ..pipeline_key(&scene.materials[d.material], topology) };
            self.pipeline(key);
            keys.push(key);
        }
        keys
    }

    fn target(&self, label: &str, extent: wgpu::Extent3d, format: wgpu::TextureFormat, samples: u32, levels: u32) -> wgpu::Texture {
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if samples == 1 && format != DEPTH_FORMAT {
            usage |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: levels,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// One multisampled pass resolved into mip 0 of `resolve`. `clear` starts
    /// from the background; otherwise the multisampled target keeps what the
    /// previous pass drew.
    #[allow(clippy::too_many_arguments)]
    fn draw_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        msaa: &wgpu::Texture,
        resolve: &wgpu::Texture,
        depth: &wgpu::Texture,
        clear: bool,
        frame_group: &wgpu::BindGroup,
        draws: &[Draw],
        keys: &[PipelineKey],
        meshes: &[(&wgpu::Buffer, Option<&wgpu::Buffer>, bool)],
        materials: &HashMap<(usize, Pass), wgpu::BindGroup>,
    ) {
        let hdr = msaa.format() == HDR_FORMAT;
        // The background: sRGB-encoded on the canvas, linear in the
        // transmission source.
        let bg = if hdr { srgb_to_linear(BACKGROUND as f32) as f64 } else { BACKGROUND };
        let msaa_view = msaa.create_view(&Default::default());
        let resolve_view = resolve.create_view(&wgpu::TextureViewDescriptor { base_mip_level: 0, mip_level_count: Some(1), ..Default::default() });
        let depth_view = depth.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("thumbnail"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa_view,
                depth_slice: None,
                resolve_target: Some(&resolve_view),
                ops: wgpu::Operations {
                    load: if clear { wgpu::LoadOp::Clear(wgpu::Color { r: bg, g: bg, b: bg, a: 1.0 }) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: if clear { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, frame_group, &[]);
        for (d, key) in draws.iter().zip(keys) {
            let (vertices, indices, points) = meshes[d.mesh];
            pass.set_pipeline(&self.pipelines[key]);
            pass.set_bind_group(1, &materials[&(d.material, d.pass)], &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            if points {
                pass.draw(0..6, d.start..d.start + d.count);
            } else if let Some(indices) = indices {
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(d.start..d.start + d.count, 0, 0..1);
            } else {
                pass.draw(d.start..d.start + d.count, 0..1);
            }
        }
    }

    /// generateMipmap for the transmission source.
    fn mipmaps(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, levels: u32) {
        for level in 1..levels {
            let src = texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level - 1, mip_level_count: Some(1), ..Default::default() });
            let dst = texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() });
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipmap"),
                layout: &self.mip_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_clamp) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mipmap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &dst, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.mip_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn buffer(&self, contents: &[u8], usage: wgpu::BufferUsages) -> Result<wgpu::Buffer, RenderError> {
        if contents.len() as u64 > self.max_buffer {
            return Err(RenderError::Gpu("buffer too large".into()));
        }
        // An empty buffer is not allowed; pad to one vertex.
        let contents = if contents.is_empty() { &[0u8; 80][..] } else { contents };
        Ok(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage }))
    }

    fn texture(&mut self, scene: &Scene, image: usize, srgb: bool, cache: &mut HashMap<(usize, bool), wgpu::TextureView>) -> wgpu::TextureView {
        if let Some(v) = cache.get(&(image, srgb)) {
            return v.clone();
        }
        let img: &Image = &scene.images[image];
        let levels = crate::texture::mip_chain(img, srgb);
        let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        let view = create_texture(&self.device, &self.queue, "model texture", img.width, img.height, format, &levels);
        cache.insert((image, srgb), view.clone());
        view
    }

    fn material_group(&mut self, scene: &Scene, index: usize, pass: Pass, topology: Topology, cache: &mut HashMap<(usize, bool), wgpu::TextureView>) -> wgpu::BindGroup {
        let m: &Material = &scene.materials[index];
        let mut flags = 0u32;
        if m.vertex_colors {
            flags |= F_VERTEX_COLORS;
        }
        if m.flat_shading {
            flags |= F_FLAT;
        }
        match (m.side, pass) {
            (Side::Double, Pass::Normal) => flags |= F_DOUBLE_SIDED,
            // Rendered as BackSide, then FrontSide.
            (Side::Double, Pass::BackFaces) | (Side::Back, _) => flags |= F_BACK_SIDE,
            _ => {}
        }
        if !m.is_transparent() {
            flags |= F_OPAQUE;
        }
        if m.size_attenuation {
            flags |= F_SIZE_ATTENUATION;
        }
        if let AlphaMode::Mask(_) = m.alpha_mode {
            flags |= F_ALPHA_TEST;
        }
        // three culls per material side; lines and points are never culled.
        if topology == Topology::Triangles {
            flags |= match (m.side, pass) {
                (Side::Double, Pass::BackFaces) | (Side::Back, _) => F_CULL_FRONT,
                (Side::Double, Pass::FrontFaces) | (Side::Front, _) => F_CULL_BACK,
                (Side::Double, Pass::Normal) => 0,
            };
        }
        let shading = match (m.shading, topology) {
            (_, Topology::Points | Topology::Lines) => 3,
            (Shading::Standard, _) => 0,
            (Shading::Phong, _) => 1,
            (Shading::Lambert, _) => 2,
            (Shading::Basic, _) => 3,
        };
        let slots = [m.map, m.emissive_map, m.metal_rough_map, m.normal_map, m.bump_map, m.ao_map, m.alpha_map, m.specular_map, m.transmission_map, m.thickness_map];
        let srgb = [true, true, false, false, false, false, false, false, false, false];
        let mut present = 0u32;
        let mut on_uv1 = 0u32;
        let mut uv_rows = [[0.0f32; 4]; SLOTS * 2];
        let mut views = Vec::with_capacity(SLOTS);
        let mut samplers = Vec::with_capacity(SLOTS);
        for (i, slot) in slots.iter().enumerate() {
            match slot.filter(|t| t.image < scene.images.len()) {
                Some(t) => {
                    present |= 1 << i;
                    if t.uv_set == 1 {
                        on_uv1 |= 1 << i;
                    }
                    let r = t.transform.0;
                    uv_rows[i * 2] = [r[0], r[1], r[2], 0.0];
                    uv_rows[i * 2 + 1] = [r[3], r[4], r[5], 0.0];
                    views.push(self.texture(scene, t.image, srgb[i], cache));
                    samplers.push(self.sampler(t.wrap_u, t.wrap_v));
                }
                None => {
                    uv_rows[i * 2] = [1.0, 0.0, 0.0, 0.0];
                    uv_rows[i * 2 + 1] = [0.0, 1.0, 0.0, 0.0];
                    views.push(self.white.clone());
                    samplers.push(self.sampler(Wrap::Repeat, Wrap::Repeat));
                }
            }
        }
        let alpha_test = if let AlphaMode::Mask(c) = m.alpha_mode { c } else { 0.0 };
        let attenuation_distance = if m.attenuation_distance.is_finite() { m.attenuation_distance } else { f32::MAX };
        let uniform = MaterialUniform {
            color: [m.color.x, m.color.y, m.color.z, m.opacity],
            emissive: [m.emissive.x, m.emissive.y, m.emissive.z, 0.0],
            specular: [m.specular.x, m.specular.y, m.specular.z, m.shininess.max(1e-4)],
            pbr: [m.roughness, m.metalness, m.ao_intensity, alpha_test],
            extra: [m.normal_scale.x, m.normal_scale.y, m.bump_scale, m.point_size],
            flags: [shading, flags, present, on_uv1],
            transmission: [m.transmission, m.ior, m.thickness, attenuation_distance],
            attenuation: [m.attenuation_color.x, m.attenuation_color.y, m.attenuation_color.z, 0.0],
            uv_rows,
        };
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("material"), contents: bytemuck::bytes_of(&uniform), usage: wgpu::BufferUsages::UNIFORM });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }];
        for (i, v) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: 1 + i as u32, resource: wgpu::BindingResource::TextureView(v) });
        }
        for (i, s) in samplers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: 1 + SLOTS as u32 + i as u32, resource: wgpu::BindingResource::Sampler(s) });
        }
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("material"), layout: &self.material_layout, entries: &entries })
    }
}

/// GPU resources of one render, shared by its passes.
#[derive(Default)]
struct Cache {
    textures: HashMap<(usize, bool), wgpu::TextureView>,
    groups: HashMap<(usize, Pass), wgpu::BindGroup>,
}

fn pipeline_key(m: &Material, topology: Topology) -> PipelineKey {
    let topology_id = match topology {
        Topology::Triangles => 0,
        Topology::Lines => 1,
        Topology::Points => 2,
    };
    let blend = if !m.is_transparent() {
        Blend::None
    } else if m.additive {
        Blend::Additive
    } else {
        Blend::Normal
    };
    PipelineKey { topology: topology_id, transmission: m.is_transmissive() && topology == Topology::Triangles, blend, depth_write: m.depth_write, hdr: false }
}

/// Object3D.lookAt for a camera: right-handed, looking down -Z.
fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        glam::Vec4::new(s.x, u.x, -f.x, 0.0),
        glam::Vec4::new(s.y, u.y, -f.y, 0.0),
        glam::Vec4::new(s.z, u.z, -f.z, 0.0),
        glam::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// PerspectiveCamera projection with WebGPU's 0..1 depth range (three's is
/// -1..1; the same planes clip, only depth precision differs).
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    let r = far / (near - far);
    Mat4::from_cols(
        glam::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, f, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, r, -1.0),
        glam::Vec4::new(0.0, 0.0, r * near, 0.0),
    )
}

/// Thumbnail.svelte isFrameEmpty(): every 4th pixel within 5 levels of the
/// first one means nothing ended up in view.
fn is_frame_empty(pixels: &[u8], size: u32) -> bool {
    let first = [pixels[0] as i32, pixels[1] as i32, pixels[2] as i32];
    let mut y = 0;
    while y < size {
        let mut x = 0;
        while x < size {
            let i = ((y * size + x) * 4) as usize;
            for c in 0..3 {
                if (pixels[i + c] as i32 - first[c]).abs() > PROBE_TOLERANCE {
                    return false;
                }
            }
            x += PROBE_STEP;
        }
        y += PROBE_STEP;
    }
    true
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

/// A 2D texture with the given mip levels (level 0 first).
fn create_texture(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, width: u32, height: u32, format: wgpu::TextureFormat, levels: &[(u32, u32, Vec<u8>)]) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes_per_texel = format.block_copy_size(None).unwrap_or(4);
    for (level, (w, h, data)) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bytes_per_texel), rows_per_image: Some(*h) },
            wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
        );
    }
    texture.create_view(&Default::default())
}
