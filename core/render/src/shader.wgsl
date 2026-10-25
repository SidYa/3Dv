// Port of the three.js r186 shaders the viewer draws models with:
// MeshStandardMaterial, MeshPhongMaterial, MeshLambertMaterial and the unlit
// MeshBasic / LineBasic / Points materials, lit by the scene of
// app/src/lib/scene.ts (ambient + three directional lights + RoomEnvironment
// PMREM), Neutral tone mapping and sRGB output.
//
// Functions keep three's names and formulas. Derivatives are taken the GLSL
// way: WGSL's dpdy runs down the framebuffer, GLSL's dFdy up, hence dfdy().

const PI: f32 = 3.141592653589793;
const RECIPROCAL_PI: f32 = 0.3183098861837907;
const EPSILON: f32 = 1e-6;

struct Frame {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    // Toward the light, view space.
    light_dir: array<vec4<f32>, 3>,
    // color * intensity
    light_color: array<vec4<f32>, 3>,
    ambient: vec4<f32>,
    // x: environment intensity, y: viewport height, z: viewport width
    params: vec4<f32>,
    // x: 1 when rendering the linear HDR transmission source (no tone
    // mapping, no sRGB encoding)
    params2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var env_map: texture_2d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;
@group(0) @binding(3) var dfg_lut: texture_2d<f32>;
// The opaque scene for transmissive materials, mipmapped.
@group(0) @binding(4) var transmission_map: texture_2d<f32>;
@group(0) @binding(5) var transmission_sampler: sampler;

// Shading models.
const STANDARD: u32 = 0u;
const PHONG: u32 = 1u;
const LAMBERT: u32 = 2u;
const BASIC: u32 = 3u;

// flags.y
const F_VERTEX_COLORS: u32 = 1u;
const F_FLAT: u32 = 2u;
const F_DOUBLE_SIDED: u32 = 4u;
const F_BACK_SIDE: u32 = 8u;
const F_OPAQUE: u32 = 16u;
const F_SIZE_ATTENUATION: u32 = 32u;
const F_ALPHA_TEST: u32 = 64u;
const F_NO_TONE_MAPPING: u32 = 128u;

// Texture slots, flags.z bit per slot present, flags.w bit per slot on uv1.
const T_MAP: u32 = 0u;
const T_EMISSIVE: u32 = 1u;
const T_METAL_ROUGH: u32 = 2u;
const T_NORMAL: u32 = 3u;
const T_BUMP: u32 = 4u;
const T_AO: u32 = 5u;
const T_ALPHA: u32 = 6u;
const T_SPECULAR: u32 = 7u;
const T_TRANSMISSION: u32 = 8u;
const T_THICKNESS: u32 = 9u;

struct Material {
    // rgb linear, a opacity
    color: vec4<f32>,
    emissive: vec4<f32>,
    // rgb specular, w shininess
    specular: vec4<f32>,
    // x roughness, y metalness, z aoMapIntensity, w alphaTest
    pbr: vec4<f32>,
    // xy normalScale, z bumpScale, w point size
    extra: vec4<f32>,
    // x shading model, y flags, z textures present, w textures on uv1
    flags: vec4<u32>,
    // x transmission, y ior, z thickness, w attenuationDistance
    transmission: vec4<f32>,
    // rgb attenuationColor
    attenuation: vec4<f32>,
    // Per slot: two rows of the 2x3 uv transform.
    uv_rows: array<vec4<f32>, 20>,
};

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var t_map: texture_2d<f32>;
@group(1) @binding(2) var t_emissive: texture_2d<f32>;
@group(1) @binding(3) var t_metal_rough: texture_2d<f32>;
@group(1) @binding(4) var t_normal: texture_2d<f32>;
@group(1) @binding(5) var t_bump: texture_2d<f32>;
@group(1) @binding(6) var t_ao: texture_2d<f32>;
@group(1) @binding(7) var t_alpha: texture_2d<f32>;
@group(1) @binding(8) var t_specular: texture_2d<f32>;
@group(1) @binding(9) var t_transmission: texture_2d<f32>;
@group(1) @binding(10) var t_thickness: texture_2d<f32>;
@group(1) @binding(11) var s_map: sampler;
@group(1) @binding(12) var s_emissive: sampler;
@group(1) @binding(13) var s_metal_rough: sampler;
@group(1) @binding(14) var s_normal: sampler;
@group(1) @binding(15) var s_bump: sampler;
@group(1) @binding(16) var s_ao: sampler;
@group(1) @binding(17) var s_alpha: sampler;
@group(1) @binding(18) var s_specular: sampler;
@group(1) @binding(19) var s_transmission: sampler;
@group(1) @binding(20) var s_thickness: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) color: vec4<f32>,
    // w = 0: no tangent attribute.
    @location(5) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // vViewPosition = -mvPosition.xyz
    @location(0) view_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) tangent: vec4<f32>,
};

fn has_flag(f: u32) -> bool {
    return (material.flags.y & f) != 0u;
}

fn has_texture(slot: u32) -> bool {
    return (material.flags.z & (1u << slot)) != 0u;
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var out: VsOut;
    let mv = frame.view * vec4<f32>(v.position, 1.0);
    out.clip = frame.proj * mv;
    out.view_position = -mv.xyz;
    var n = (frame.view * vec4<f32>(v.normal, 0.0)).xyz;
    // FLIP_SIDED
    if has_flag(F_BACK_SIDE) {
        n = -n;
    }
    out.normal = n;
    out.uv0 = v.uv0;
    out.uv1 = v.uv1;
    out.color = v.color;
    out.tangent = vec4<f32>((frame.view * vec4<f32>(v.tangent.xyz, 0.0)).xyz, v.tangent.w);
    return out;
}

// Points: one screen-aligned square per vertex, sized like gl_PointSize in
// three's points.glsl (size * scale / -mvPosition.z with sizeAttenuation).
struct PointIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_points(p: PointIn, @builtin(vertex_index) corner: u32) -> VsOut {
    var out: VsOut;
    let mv = frame.view * vec4<f32>(p.position, 1.0);
    var clip = frame.proj * mv;
    var size = material.extra.w;
    if has_flag(F_SIZE_ATTENUATION) {
        size = size * (frame.params.y * 0.5 / -mv.z);
    }
    // Hardware points never get smaller than one pixel.
    size = max(size, 1.0);
    let offsets = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.5),
    );
    let o = offsets[corner % 6u];
    clip = vec4<f32>(clip.xy + o * size * 2.0 / frame.params.zy * clip.w, clip.zw);
    out.clip = clip;
    out.view_position = -mv.xyz;
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.uv0 = o + vec2<f32>(0.5);
    out.uv1 = out.uv0;
    out.color = p.color;
    out.tangent = vec4<f32>(0.0);
    return out;
}

fn uv_for(slot: u32, in: VsOut) -> vec2<f32> {
    var uv = in.uv0;
    if (material.flags.w & (1u << slot)) != 0u {
        uv = in.uv1;
    }
    let r0 = material.uv_rows[slot * 2u];
    let r1 = material.uv_rows[slot * 2u + 1u];
    return vec2<f32>(dot(r0.xyz, vec3<f32>(uv, 1.0)), dot(r1.xyz, vec3<f32>(uv, 1.0)));
}

fn dfdy2(v: vec2<f32>) -> vec2<f32> {
    return -dpdy(v);
}

fn dfdy3(v: vec3<f32>) -> vec3<f32> {
    return -dpdy(v);
}

fn pow2(x: f32) -> f32 {
    return x * x;
}

fn pow4(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2;
}

fn BRDF_Lambert(diffuse_color: vec3<f32>) -> vec3<f32> {
    return RECIPROCAL_PI * diffuse_color;
}

fn F_Schlick(f0: vec3<f32>, f90: f32, dotVH: f32) -> vec3<f32> {
    let fresnel = exp2((-5.55473 * dotVH - 6.98316) * dotVH);
    return f0 * (1.0 - fresnel) + (f90 * fresnel);
}

fn V_GGX_SmithCorrelated(alpha: f32, dotNL: f32, dotNV: f32) -> f32 {
    let a2 = pow2(alpha);
    let gv = dotNL * sqrt(a2 + (1.0 - a2) * pow2(dotNV));
    let gl = dotNV * sqrt(a2 + (1.0 - a2) * pow2(dotNL));
    return 0.5 / max(gv + gl, EPSILON);
}

fn D_GGX(alpha: f32, dotNH: f32) -> f32 {
    let a2 = pow2(alpha);
    let denom = pow2(dotNH) * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2 / pow2(denom);
}

struct Physical {
    diffuse_color: vec3<f32>,
    diffuse_contribution: vec3<f32>,
    specular_color: vec3<f32>,
    specular_color_blended: vec3<f32>,
    roughness: f32,
    metalness: f32,
    specular_f90: f32,
    dfg: vec2<f32>,
    multi_scattering_compensation: vec3<f32>,
};

fn BRDF_GGX(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, m: Physical) -> vec3<f32> {
    let f0 = m.specular_color_blended;
    let alpha = pow2(m.roughness);
    let half_dir = normalize(light_dir + view_dir);
    let dotNL = saturate(dot(normal, light_dir));
    let dotNV = saturate(dot(normal, view_dir));
    let dotNH = saturate(dot(normal, half_dir));
    let dotVH = saturate(dot(view_dir, half_dir));
    let F = F_Schlick(f0, m.specular_f90, dotVH);
    let V = V_GGX_SmithCorrelated(alpha, dotNL, dotNV);
    let D = D_GGX(alpha, dotNH);
    return F * (V * D);
}

fn BRDF_BlinnPhong(light_dir: vec3<f32>, view_dir: vec3<f32>, normal: vec3<f32>, specular_color: vec3<f32>, shininess: f32) -> vec3<f32> {
    let half_dir = normalize(light_dir + view_dir);
    let dotNH = saturate(dot(normal, half_dir));
    let dotVH = saturate(dot(view_dir, half_dir));
    let F = F_Schlick(specular_color, 1.0, dotVH);
    let G = 0.25;
    let D = RECIPROCAL_PI * (shininess * 0.5 + 1.0) * pow(dotNH, shininess);
    return F * (G * D);
}

struct Scattering {
    single: vec3<f32>,
    multi: vec3<f32>,
};

fn computeMultiscattering(fab: vec2<f32>, specular_color: vec3<f32>, specular_f90: f32) -> Scattering {
    let Fr = specular_color;
    let FssEss = Fr * fab.x + specular_f90 * fab.y;
    let Ess = fab.x + fab.y;
    let Ems = 1.0 - Ess;
    let Favg = Fr + (1.0 - Fr) * 0.047619;
    let Fms = FssEss * Favg / (1.0 - Ems * Favg);
    return Scattering(FssEss, Fms * Ems);
}

fn EnvironmentBRDF(normal: vec3<f32>, view_dir: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, roughness: f32) -> vec3<f32> {
    let dotNV = saturate(dot(normal, view_dir));
    let fab = textureSampleLevel(dfg_lut, linear_clamp, vec2<f32>(roughness, dotNV), 0.0).rg;
    return specular_color * fab.x + specular_f90 * fab.y;
}

// --- transmission (transmission_pars_fragment) ------------------------------

fn w0(a: f32) -> f32 { return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0); }
fn w1(a: f32) -> f32 { return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0); }
fn w2(a: f32) -> f32 { return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0); }
fn w3(a: f32) -> f32 { return (1.0 / 6.0) * (a * a * a); }
fn g0(a: f32) -> f32 { return w0(a) + w1(a); }
fn g1(a: f32) -> f32 { return w2(a) + w3(a); }
fn h0(a: f32) -> f32 { return -1.0 + w1(a) / (w0(a) + w1(a)); }
fn h1(a: f32) -> f32 { return 1.0 + w3(a) / (w2(a) + w3(a)); }

// The texture rows run top-down; three's v runs bottom-up, so v is flipped
// on every fetch.
fn transmission_fetch(uv: vec2<f32>, lod: f32) -> vec4<f32> {
    return textureSampleLevel(transmission_map, transmission_sampler, vec2<f32>(uv.x, 1.0 - uv.y), lod);
}

fn bicubic(uv_in: vec2<f32>, texel_size: vec4<f32>, lod: f32) -> vec4<f32> {
    let uv = uv_in * texel_size.zw + 0.5;
    let iuv = floor(uv);
    let fuv = fract(uv);
    let g0x = g0(fuv.x);
    let g1x = g1(fuv.x);
    let h0x = h0(fuv.x);
    let h1x = h1(fuv.x);
    let h0y = h0(fuv.y);
    let h1y = h1(fuv.y);
    let p0 = (vec2<f32>(iuv.x + h0x, iuv.y + h0y) - 0.5) * texel_size.xy;
    let p1 = (vec2<f32>(iuv.x + h1x, iuv.y + h0y) - 0.5) * texel_size.xy;
    let p2 = (vec2<f32>(iuv.x + h0x, iuv.y + h1y) - 0.5) * texel_size.xy;
    let p3 = (vec2<f32>(iuv.x + h1x, iuv.y + h1y) - 0.5) * texel_size.xy;
    return g0(fuv.y) * (g0x * transmission_fetch(p0, lod) + g1x * transmission_fetch(p1, lod))
        + g1(fuv.y) * (g0x * transmission_fetch(p2, lod) + g1x * transmission_fetch(p3, lod));
}

fn textureBicubic(uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let levels = i32(textureNumLevels(transmission_map)) - 1;
    let f = clamp(i32(floor(lod)), 0, levels);
    let c = clamp(i32(ceil(lod)), 0, levels);
    let f_size = vec2<f32>(textureDimensions(transmission_map, f));
    let c_size = vec2<f32>(textureDimensions(transmission_map, c));
    let f_sample = bicubic(uv, vec4<f32>(1.0 / f_size, f_size), floor(lod));
    let c_sample = bicubic(uv, vec4<f32>(1.0 / c_size, c_size), ceil(lod));
    return mix(f_sample, c_sample, fract(lod));
}

fn applyIorToRoughness(roughness: f32, ior: f32) -> f32 {
    return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

fn volumeAttenuation(distance: f32, attenuation_color: vec3<f32>, attenuation_distance: f32) -> vec3<f32> {
    // isinf(): an infinite distance arrives as a huge value.
    if attenuation_distance > 1e30 {
        return vec3<f32>(1.0);
    }
    let coefficient = -log(attenuation_color) / attenuation_distance;
    return exp(-coefficient * distance);
}

// getIBLVolumeRefraction() in view space: the geometry is already in world
// space, so three's modelScale is 1.
fn volume_refraction(n: vec3<f32>, v: vec3<f32>, view_position: vec3<f32>, roughness: f32, diffuse_color: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, ior: f32, thickness: f32) -> vec4<f32> {
    let ray = normalize(refract(-v, n, 1.0 / ior)) * thickness;
    let exit = view_position + ray;
    let ndc = frame.proj * vec4<f32>(exit, 1.0);
    let coords = (ndc.xy / ndc.w + 1.0) / 2.0;
    let size = f32(textureDimensions(transmission_map, 0).x);
    let lod = log2(size) * applyIorToRoughness(roughness, ior);
    let transmitted_light = textureBicubic(coords, lod);
    let transmittance = diffuse_color * volumeAttenuation(length(ray), material.attenuation.rgb, material.transmission.w);
    let attenuated = transmittance * transmitted_light.rgb;
    let F = EnvironmentBRDF(n, v, specular_color, specular_f90, roughness);
    let factor = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    return vec4<f32>((1.0 - F) * attenuated, 1.0 - (1.0 - transmitted_light.a) * factor);
}

fn computeSpecularOcclusion(dotNV: f32, ambient_occlusion: f32, roughness: f32) -> f32 {
    return saturate(pow(dotNV + ambient_occlusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + ambient_occlusion);
}

// --- PMREM (cube_uv_reflection_fragment) ----------------------------------

const cubeUV_minMipLevel: f32 = 4.0;
const cubeUV_minTileSize: f32 = 16.0;
const CUBEUV_MAX_MIP: f32 = 8.0;

fn getFace(direction: vec3<f32>) -> f32 {
    let a = abs(direction);
    var face = -1.0;
    if a.x > a.z {
        if a.x > a.y {
            face = select(3.0, 0.0, direction.x > 0.0);
        } else {
            face = select(4.0, 1.0, direction.y > 0.0);
        }
    } else {
        if a.z > a.y {
            face = select(5.0, 2.0, direction.z > 0.0);
        } else {
            face = select(4.0, 1.0, direction.y > 0.0);
        }
    }
    return face;
}

fn getUV(direction: vec3<f32>, face: f32) -> vec2<f32> {
    var uv: vec2<f32>;
    if face == 0.0 {
        uv = vec2<f32>(direction.z, direction.y) / abs(direction.x);
    } else if face == 1.0 {
        uv = vec2<f32>(-direction.x, -direction.z) / abs(direction.y);
    } else if face == 2.0 {
        uv = vec2<f32>(-direction.x, direction.y) / abs(direction.z);
    } else if face == 3.0 {
        uv = vec2<f32>(-direction.z, direction.y) / abs(direction.x);
    } else if face == 4.0 {
        uv = vec2<f32>(-direction.x, direction.z) / abs(direction.y);
    } else {
        uv = vec2<f32>(direction.x, direction.y) / abs(direction.z);
    }
    return 0.5 * (uv + 1.0);
}

fn bilinearCubeUV(direction: vec3<f32>, mip_in: f32) -> vec3<f32> {
    var face = getFace(direction);
    let filterInt = max(cubeUV_minMipLevel - mip_in, 0.0);
    let mipInt = max(mip_in, cubeUV_minMipLevel);
    let faceSize = exp2(mipInt);
    var uv = getUV(direction, face) * (faceSize - 2.0) + 1.0;
    if face > 2.0 {
        uv.y += faceSize;
        face -= 3.0;
    }
    uv.x += face * faceSize;
    uv.x += filterInt * 3.0 * cubeUV_minTileSize;
    uv.y += 4.0 * (exp2(CUBEUV_MAX_MIP) - faceSize);
    let size = vec2<f32>(textureDimensions(env_map));
    uv = uv / size;
    // The baked atlas is stored top row first; GL's v runs bottom-up.
    return textureSampleLevel(env_map, linear_clamp, vec2<f32>(uv.x, 1.0 - uv.y), 0.0).rgb;
}

fn roughnessToMip(roughness: f32) -> f32 {
    var mip = 0.0;
    if roughness >= 0.8 {
        mip = (1.0 - roughness) * (-1.0 - -2.0) / (1.0 - 0.8) + -2.0;
    } else if roughness >= 0.4 {
        mip = (0.8 - roughness) * (2.0 - -1.0) / (0.8 - 0.4) + -1.0;
    } else if roughness >= 0.305 {
        mip = (0.4 - roughness) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0;
    } else if roughness >= 0.21 {
        mip = (0.305 - roughness) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0;
    } else {
        mip = -2.0 * log2(1.16 * roughness);
    }
    return mip;
}

fn textureCubeUV(sample_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    let mip = clamp(roughnessToMip(roughness), -2.0, CUBEUV_MAX_MIP);
    let mipF = fract(mip);
    let mipInt = floor(mip);
    let color0 = bilinearCubeUV(sample_dir, mipInt);
    if mipF == 0.0 {
        return color0;
    }
    let color1 = bilinearCubeUV(sample_dir, mipInt + 1.0);
    return mix(color0, color1, mipF);
}

// Inverse view rotation (the view matrix is orthonormal).
fn to_world(dir: vec3<f32>) -> vec3<f32> {
    return normalize((vec4<f32>(dir, 0.0) * frame.view).xyz);
}

fn getIBLIrradiance(normal: vec3<f32>) -> vec3<f32> {
    return PI * textureCubeUV(to_world(normal), 1.0) * frame.params.x;
}

fn getIBLRadiance(view_dir: vec3<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    var reflect_vec = reflect(-view_dir, normal);
    reflect_vec = normalize(mix(reflect_vec, normal, pow4(roughness)));
    return textureCubeUV(to_world(reflect_vec), roughness) * frame.params.x;
}

// --- output ---------------------------------------------------------------

fn NeutralToneMapping(c: vec3<f32>) -> vec3<f32> {
    let StartCompression = 0.8 - 0.04;
    let Desaturation = 0.15;
    var color = c;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    color -= offset;
    let peak = max(color.r, max(color.g, color.b));
    if peak < StartCompression {
        return color;
    }
    let d = 1.0 - StartCompression;
    let newPeak = 1.0 - d * d / (peak + d - StartCompression);
    color *= newPeak / peak;
    let g = 1.0 - 1.0 / (Desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3<f32>(newPeak), g);
}

fn sRGBTransferOETF(v: vec3<f32>) -> vec3<f32> {
    let lo = v * 12.92;
    let hi = pow(max(v, vec3<f32>(0.0)), vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);
    return select(hi, lo, v <= vec3<f32>(0.0031308));
}

fn finish(outgoing: vec3<f32>, alpha: f32) -> vec4<f32> {
    // The transmission source is linear and not tone mapped.
    if frame.params2.x > 0.5 {
        return vec4<f32>(outgoing, alpha);
    }
    var color = outgoing;
    if !has_flag(F_NO_TONE_MAPPING) {
        color = NeutralToneMapping(color);
    }
    // The canvas is 8-bit sRGB-encoded; blending and multisample resolve
    // happen on encoded values, as in WebGL.
    return vec4<f32>(sRGBTransferOETF(color), alpha);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    // All texture reads up front, in uniform control flow (missing
    // textures are a white texel).
    let uv_map = uv_for(T_MAP, in);
    let uv_bump = uv_for(T_BUMP, in);
    let uv_normal = uv_for(T_NORMAL, in);
    let texel_map = textureSample(t_map, s_map, uv_map);
    let texel_emissive = textureSample(t_emissive, s_emissive, uv_for(T_EMISSIVE, in));
    let texel_mr = textureSample(t_metal_rough, s_metal_rough, uv_for(T_METAL_ROUGH, in));
    let texel_normal = textureSample(t_normal, s_normal, uv_normal);
    let texel_ao = textureSample(t_ao, s_ao, uv_for(T_AO, in));
    let texel_alpha = textureSample(t_alpha, s_alpha, uv_for(T_ALPHA, in));
    let texel_specular = textureSample(t_specular, s_specular, uv_for(T_SPECULAR, in));
    let texel_transmission = textureSample(t_transmission, s_transmission, uv_for(T_TRANSMISSION, in));
    let texel_thickness = textureSample(t_thickness, s_thickness, uv_for(T_THICKNESS, in));
    // dHdxy_fwd()
    let dSTdx = dpdx(uv_bump);
    let dSTdy = dfdy2(uv_bump);
    let bump_scale = material.extra.z;
    let Hll = bump_scale * textureSample(t_bump, s_bump, uv_bump).x;
    let dBx = bump_scale * textureSample(t_bump, s_bump, uv_bump + dSTdx).x - Hll;
    let dBy = bump_scale * textureSample(t_bump, s_bump, uv_bump + dSTdy).x - Hll;
    let q0 = dpdx(in.view_position);
    let q1 = dfdy3(in.view_position);
    let st0 = dpdx(uv_normal);
    let st1 = dfdy2(uv_normal);

    let model = material.flags.x;
    var diffuse_color = material.color;
    if has_texture(T_MAP) {
        diffuse_color *= texel_map;
    }
    if has_flag(F_VERTEX_COLORS) {
        diffuse_color *= in.color;
    }
    if has_texture(T_ALPHA) {
        diffuse_color.a *= texel_alpha.g;
    }
    if has_flag(F_ALPHA_TEST) && diffuse_color.a < material.pbr.w {
        discard;
    }
    var alpha = diffuse_color.a;
    if has_flag(F_OPAQUE) {
        alpha = 1.0;
    }

    if model == BASIC {
        var outgoing = diffuse_color.rgb;
        if has_texture(T_AO) {
            outgoing *= (texel_ao.r - 1.0) * material.pbr.z + 1.0;
        }
        return finish(outgoing, alpha);
    }

    // normal_fragment_begin
    let face_direction = select(-1.0, 1.0, front_facing);
    var normal: vec3<f32>;
    if has_flag(F_FLAT) {
        normal = normalize(cross(q0, q1));
    } else {
        normal = normalize(in.normal);
        if has_flag(F_DOUBLE_SIDED) {
            normal *= face_direction;
        }
    }
    let non_perturbed_normal = normal;

    // normal_fragment_maps (tangent space) or bump mapping
    if has_texture(T_NORMAL) {
        var tbn: mat3x3<f32>;
        if in.tangent.w != 0.0 && !has_flag(F_FLAT) {
            let t = normalize(in.tangent.xyz);
            var b = normalize(cross(normalize(in.normal), t) * in.tangent.w);
            if has_flag(F_BACK_SIDE) {
                b = -b;
            }
            tbn = mat3x3<f32>(t, b, normal);
        } else {
            // getTangentFrame()
            let N = normal;
            let q1perp = cross(q1, N);
            let q0perp = cross(N, q0);
            let T = q1perp * st0.x + q0perp * st1.x;
            let B = q1perp * st0.y + q0perp * st1.y;
            let det = max(dot(T, T), dot(B, B));
            let scale = select(inverseSqrt(det), 0.0, det == 0.0);
            tbn = mat3x3<f32>(T * scale, B * scale, N);
        }
        if has_flag(F_DOUBLE_SIDED) {
            tbn[0] *= face_direction;
            tbn[1] *= face_direction;
        }
        var mapN = texel_normal.xyz * 2.0 - 1.0;
        mapN = vec3<f32>(mapN.xy * material.extra.xy, mapN.z);
        normal = normalize(tbn * mapN);
    } else if has_texture(T_BUMP) {
        // perturbNormalArb()
        let vSigmaX = normalize(q0);
        let vSigmaY = normalize(q1);
        let vN = normal;
        let R1 = cross(vSigmaY, vN);
        let R2 = cross(vN, vSigmaX);
        let fDet = dot(vSigmaX, R1) * face_direction;
        let vGrad = sign(fDet) * (dBx * R1 + dBy * R2);
        normal = normalize(abs(fDet) * normal - vGrad);
    }

    var emissive = material.emissive.rgb;
    if has_texture(T_EMISSIVE) {
        emissive *= texel_emissive.rgb;
    }

    let geometry_normal = normal;
    let view_dir = normalize(in.view_position);
    var ambient_occlusion = 1.0;
    if has_texture(T_AO) {
        ambient_occlusion = (texel_ao.r - 1.0) * material.pbr.z + 1.0;
    }

    var direct_diffuse = vec3<f32>(0.0);
    var direct_specular = vec3<f32>(0.0);
    var indirect_diffuse = vec3<f32>(0.0);
    var indirect_specular = vec3<f32>(0.0);

    if model == STANDARD {
        var roughness_factor = material.pbr.x;
        var metalness_factor = material.pbr.y;
        if has_texture(T_METAL_ROUGH) {
            roughness_factor *= texel_mr.g;
            metalness_factor *= texel_mr.b;
        }
        var m: Physical;
        m.diffuse_color = diffuse_color.rgb;
        m.diffuse_contribution = diffuse_color.rgb * (1.0 - metalness_factor);
        m.metalness = metalness_factor;
        let dxy = max(abs(dpdx(non_perturbed_normal)), abs(dpdy(non_perturbed_normal)));
        let geometry_roughness = max(max(dxy.x, dxy.y), dxy.z);
        m.roughness = max(roughness_factor, 0.0525);
        m.roughness += geometry_roughness;
        m.roughness = min(m.roughness, 1.0);
        m.specular_color = vec3<f32>(0.04);
        m.specular_color_blended = mix(m.specular_color, diffuse_color.rgb, metalness_factor);
        m.specular_f90 = 1.0;
        let dotNVms = saturate(dot(geometry_normal, view_dir));
        m.dfg = textureSampleLevel(dfg_lut, linear_clamp, vec2<f32>(m.roughness, dotNVms), 0.0).rg;
        let EssMs = m.dfg.x + m.dfg.y;
        m.multi_scattering_compensation = 1.0 + m.specular_color_blended * (1.0 / EssMs - 1.0);

        for (var i = 0; i < 3; i++) {
            let L = frame.light_dir[i].xyz;
            let dotNL = saturate(dot(geometry_normal, L));
            let irradiance = dotNL * frame.light_color[i].rgb;
            direct_specular += irradiance * BRDF_GGX(L, view_dir, geometry_normal, m) * m.multi_scattering_compensation;
            let half_dir = normalize(L + view_dir);
            let dotVH = saturate(dot(view_dir, half_dir));
            let F = F_Schlick(m.specular_color, m.specular_f90, dotVH);
            direct_diffuse += irradiance * BRDF_Lambert(m.diffuse_contribution) * (1.0 - F);
        }

        // RE_IndirectDiffuse_Physical with the ambient light
        let irradiance = frame.ambient.rgb;
        let s0 = computeMultiscattering(m.dfg, m.specular_color, m.specular_f90);
        indirect_diffuse += irradiance * BRDF_Lambert(m.diffuse_contribution) * (1.0 - s0.single - s0.multi);

        // RE_IndirectSpecular_Physical with the environment
        let ibl_irradiance = getIBLIrradiance(geometry_normal);
        let radiance = getIBLRadiance(view_dir, geometry_normal, m.roughness);
        let sd = computeMultiscattering(m.dfg, m.specular_color, m.specular_f90);
        let sm = computeMultiscattering(m.dfg, m.diffuse_color, m.specular_f90);
        let single_scattering = mix(sd.single, sm.single, m.metalness);
        let multi_scattering = mix(sd.multi, sm.multi, m.metalness);
        let total_scattering_dielectric = sd.single + sd.multi;
        let diffuse = m.diffuse_contribution * (1.0 - total_scattering_dielectric);
        let cosine_weighted_irradiance = ibl_irradiance * RECIPROCAL_PI;
        indirect_specular += radiance * single_scattering + multi_scattering * cosine_weighted_irradiance;
        indirect_diffuse += diffuse * cosine_weighted_irradiance;

        // aomap_fragment
        if has_texture(T_AO) {
            indirect_diffuse *= ambient_occlusion;
            let dotNV = saturate(dot(geometry_normal, view_dir));
            indirect_specular *= computeSpecularOcclusion(dotNV, ambient_occlusion, m.roughness);
        }

        // transmission_fragment: the diffuse part gives way to the light
        // refracted through the surface.
        var transmission = material.transmission.x;
        if transmission > 0.0 {
            if has_texture(T_TRANSMISSION) {
                transmission *= texel_transmission.r;
            }
            var thickness = material.transmission.z;
            if has_texture(T_THICKNESS) {
                thickness *= texel_thickness.g;
            }
            let transmitted = volume_refraction(geometry_normal, view_dir, -in.view_position, m.roughness, m.diffuse_contribution, m.specular_color_blended, m.specular_f90, material.transmission.y, thickness);
            direct_diffuse = mix(direct_diffuse + indirect_diffuse, transmitted.rgb, transmission);
            indirect_diffuse = vec3<f32>(0.0);
            alpha *= mix(1.0, transmitted.a, transmission);
            if has_flag(F_OPAQUE) {
                alpha = 1.0;
            }
        }
    } else {
        // Phong and Lambert
        var specular_strength = 1.0;
        if has_texture(T_SPECULAR) {
            specular_strength = texel_specular.r;
        }
        for (var i = 0; i < 3; i++) {
            let L = frame.light_dir[i].xyz;
            let dotNL = saturate(dot(geometry_normal, L));
            let irradiance = dotNL * frame.light_color[i].rgb;
            direct_diffuse += irradiance * BRDF_Lambert(diffuse_color.rgb);
            if model == PHONG {
                direct_specular += irradiance * BRDF_BlinnPhong(L, view_dir, geometry_normal, material.specular.rgb, material.specular.w) * specular_strength;
            }
        }
        // lights_fragment_end: ambient plus the environment's irradiance.
        let irradiance = frame.ambient.rgb + getIBLIrradiance(geometry_normal);
        indirect_diffuse += irradiance * BRDF_Lambert(diffuse_color.rgb);
        if has_texture(T_AO) {
            indirect_diffuse *= ambient_occlusion;
        }
    }

    let outgoing = direct_diffuse + indirect_diffuse + direct_specular + indirect_specular + emissive;
    return finish(outgoing, alpha);
}
