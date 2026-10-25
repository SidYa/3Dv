// The lighting data baked from three.js by tools/bake-env.mjs.

/// RoomEnvironment prefiltered by PMREMGenerator, in three's CubeUV atlas
/// layout: (width, height, RGBA16F texels as half-float bits, top row first).
pub fn room_pmrem() -> (u32, u32, Vec<u16>) {
    let packed = include_bytes!("../assets/room_pmrem.rgbe.z");
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(packed).expect("baked environment is valid zlib");
    let width = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let height = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let rgbe = &raw[8..];
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for px in rgbe.chunks_exact(4) {
        let (r, g, b, e) = (px[0], px[1], px[2], px[3]);
        if e == 0 {
            out.extend_from_slice(&[0, 0, 0, f32_to_f16(1.0)]);
            continue;
        }
        let scale = 2f32.powi(e as i32 - 128) / 256.0;
        out.extend_from_slice(&[f32_to_f16(r as f32 * scale), f32_to_f16(g as f32 * scale), f32_to_f16(b as f32 * scale), f32_to_f16(1.0)]);
    }
    (width, height, out)
}

/// three's DFG LUT: 16x16 RG16F.
pub fn dfg_lut() -> Vec<u16> {
    include_bytes!("../assets/dfg_lut.bin").chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
}

/// IEEE 754 binary16 from f32, round to nearest even; values beyond the
/// half range saturate to infinity.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mant + 0xfff + ((mant >> 13) & 1);
    if rounded & 0x80_0000 != 0 {
        let e = e + 1;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((e as u16) << 10);
    }
    sign | ((e as u16) << 10) | (rounded >> 13) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
    }

    #[test]
    fn assets_decode() {
        let (w, h, texels) = room_pmrem();
        assert_eq!((w, h), (768, 1024));
        assert_eq!(texels.len(), (w * h * 4) as usize);
        assert_eq!(dfg_lut().len(), 512);
    }
}
