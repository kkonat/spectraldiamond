struct Params {
    wnorm : vec4<f32>,
    dims  : vec4<u32>,
    cfg   : vec4<f32>,
    cfg2  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> out_px : array<vec4<f32>>;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

// ------------------------------------------------------ spectral -> colour
fn cie_g(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let s = select(s2, s1, x < mu);
    let t = (x - mu) / s;
    return exp(-0.5 * t * t);
}

fn spectral_weight(lam: f32) -> vec3<f32> {
    let X =  1.056 * cie_g(lam, 599.8, 37.9, 31.0)
           + 0.362 * cie_g(lam, 442.0, 16.0, 26.7)
           - 0.065 * cie_g(lam, 501.1, 20.4, 26.2);
    let Y =  0.821 * cie_g(lam, 568.8, 46.9, 40.5)
           + 0.286 * cie_g(lam, 530.9, 16.3, 31.1);
    let Z =  1.217 * cie_g(lam, 437.0, 11.8, 36.0)
           + 0.681 * cie_g(lam, 459.0, 26.0, 13.8);
    let rgb = vec3<f32>( 3.2406 * X - 1.5372 * Y - 0.4986 * Z,
                        -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
                         0.0557 * X - 0.2040 * Y + 1.0570 * Z);
    return rgb / P.wnorm.xyz;
}

// ---------------------------------------------------------- stage 3: curve
fn aces(x: f32) -> f32 {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

fn aces_per_channel(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(aces(c.r), aces(c.g), aces(c.b));
}

fn tonemap_luminance(c: vec3<f32>, strength: f32) -> vec3<f32> {
    let L  = dot(c, LUMA);
    let Lt = aces(max(L, 0.0));
    let lum_path = select(vec3<f32>(0.0), c * (Lt / L), L > 1e-6);
    return mix(aces_per_channel(max(c, vec3<f32>(0.0))), lum_path, strength);
}

// ---------------------------------------------------- stage 4: Oklab chroma
fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let lms = vec3<f32>(dot(c, vec3<f32>(0.4122214708, 0.5363325363, 0.0514459929)),
                        dot(c, vec3<f32>(0.2119034982, 0.6806995451, 0.1073969566)),
                        dot(c, vec3<f32>(0.0883024619, 0.2817188376, 0.6299787005)));
    let r = sign(lms) * pow(abs(lms), vec3<f32>(1.0 / 3.0));
    return vec3<f32>(dot(r, vec3<f32>( 0.2104542553,  0.7936177850, -0.0040720468)),
                     dot(r, vec3<f32>( 1.9779984951, -2.4285922050,  0.4505937099)),
                     dot(r, vec3<f32>( 0.0259040371,  0.7827717662, -0.8086757660)));
}

fn oklab_to_linear(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_; let m = m_ * m_ * m_; let s = s_ * s_ * s_;
    return vec3<f32>( 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                     -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                     -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

fn boost_chroma(c: vec3<f32>, amount: f32) -> vec3<f32> {
    let lab = linear_to_oklab(max(c, vec3<f32>(0.0)));
    let C   = length(lab.yz);
    let k   = 1.0 + (amount - 1.0) / (1.0 + 4.0 * C);
    return oklab_to_linear(vec3<f32>(lab.x, lab.yz * k));
}

// ------------------------------------------------------ stage 5: gamut fit
// Spec stage 5(a): desaturate the minimum necessary to clear negatives, at
// constant hue and constant luminance. Homogeneous in c, so it commutes with
// any exposure scale applied before it.
fn clear_negatives(c: vec3<f32>) -> vec3<f32> {
    let L  = max(dot(c, LUMA), 0.0);
    let mn = min(min(c.r, c.g), c.b);
    let t  = select(1.0, clamp(L / max(L - mn, 1e-9), 0.0, 1.0), mn < 0.0);
    return vec3<f32>(L) + t * (c - vec3<f32>(L));
}

fn desaturate_to_white(c: vec3<f32>) -> vec3<f32> {
    let L = clamp(dot(c, LUMA), 0.0, 1.0);
    let d = c - vec3<f32>(L);
    let hi = mix(vec3<f32>(1e9), (vec3<f32>(1.0) - L) / max(d, vec3<f32>( 1e-9)),
                 step(vec3<f32>(1e-9), d));
    let lo = mix(vec3<f32>(1e9), (vec3<f32>(0.0) - L) / min(d, vec3<f32>(-1e-9)),
                 step(d, vec3<f32>(-1e-9)));
    let t = min(1.0, min(min(min(hi.x, hi.y), hi.z), min(min(lo.x, lo.y), lo.z)));
    return clamp(vec3<f32>(L) + max(t, 0.0) * d, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn fit_gamut(col: vec3<f32>, keep_chroma: f32) -> vec3<f32> {
    let c = clear_negatives(col);
    let chroma_path = c / max(max(max(c.r, c.g), c.b), 1.0);
    if (keep_chroma >= 1.0) { return clamp(chroma_path, vec3<f32>(0.0), vec3<f32>(1.0)); }
    return clamp(mix(desaturate_to_white(c), chroma_path, keep_chroma),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// Brightest in-gamut colour of this chromaticity: clear negatives, then scale
// so the largest channel lands exactly on 1.0. Unlike fit_gamut this divides
// unconditionally, so it brightens as well as darkens -- that is the "maximum
// lightness without clipping" the swatch is for.
fn max_out(col: vec3<f32>) -> vec3<f32> {
    let c  = clear_negatives(col);
    let mx = max(max(c.r, c.g), c.b);
    return clamp(c / max(mx, 1e-9), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ----------------------------------------------------------- stage 7: OETF
fn srgb_oetf(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// ------------------------------------------------- display-gamut reference
// The most saturated thing the monitor can emit at this hue: push the display
// triple to the gamut corner, min channel to 0 and max channel to 1, HSV hue
// held. Nothing outside this is reproducible, so the strip is the ceiling the
// band above it is being measured against.
//
// Done on the sRGB-encoded triple, which is where HSV is conventionally
// defined. Note the caveat from the spec: HSV hue is not perceptual hue, so
// the corner sits at a slightly different Oklab hue than the sample it came
// from -- fine for reading off headroom, not a colour you should grade toward.
fn gamut_corner(srgb: vec3<f32>) -> vec3<f32> {
    let mn = min(min(srgb.r, srgb.g), srgb.b);
    let mx = max(max(srgb.r, srgb.g), srgb.b);
    return select(vec3<f32>(mx), (srgb - mn) / (mx - mn), (mx - mn) > 1e-6);
}

fn hsv_hue(h: f32) -> vec3<f32> {                  // s = v = 1, h in degrees
    let k = (h / 60.0) % 6.0;
    return clamp(vec3<f32>(abs(k - 3.0) - 1.0,
                           2.0 - abs(k - 2.0),
                           2.0 - abs(k - 4.0)), vec3<f32>(0.0), vec3<f32>(1.0));
}

// --------------------------------------------------------------- the bands
fn band_colour(band: u32, lam: f32) -> vec3<f32> {
    let hdr   = spectral_weight(lam);
    let boost = P.cfg.z;
    let keep  = P.cfg.w;
    let expo  = P.cfg2.x;

    // 0: maximum saturation and lightness, chromaticity untouched.
    if (band == 0u) { return max_out(hdr); }

    // 1: same, with the Oklab vibrance pass, re-maxed so the comparison is
    //    lightness-for-lightness and only the chroma differs.
    if (band == 1u) { return max_out(boost_chroma(max_out(hdr), boost)); }

    // 2: the shipping grade -- luminance tone curve, boost, fit at keepChroma.
    if (band == 2u) {
        let c = tonemap_luminance(max_out(hdr) * expo, 1.0);
        return fit_gamut(boost_chroma(c, boost), keep);
    }

    // 3: the old look -- per-channel ACES on a clamped colour.
    return aces_per_channel(max(max_out(hdr) * expo, vec3<f32>(0.0)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.dims.x;
    let h = P.dims.y;
    if (gid.x >= w || gid.y >= h) { return; }

    let bh    = P.dims.w;
    let nband = P.dims.z;
    let band  = min(gid.y / bh, nband - 1u);
    let sep   = u32(P.cfg2.y);
    let sub   = u32(P.cfg2.z);          // height of the reference strip
    let mode  = u32(P.cfg2.w);          // 0 = hue-matched corner, 1 = hue sweep
    let y0    = gid.y - band * bh;

    let t   = (f32(gid.x) + 0.5) / f32(w);
    let lam = P.cfg.x + t * P.cfg.y;

    var rgb: vec3<f32>;
    if ((band > 0u && y0 < sep) || (sub > 0u && y0 >= bh - sub && y0 < bh - sub + sep)) {
        rgb = vec3<f32>(0.05, 0.05, 0.05);           // separator, sRGB
    } else if (sub > 0u && y0 >= bh - sub) {
        if (mode == 0u) {
            rgb = gamut_corner(srgb_oetf(band_colour(band, lam)));
        } else {
            rgb = hsv_hue(mix(300.0, 0.0, t));       // plain full-RGB rainbow
        }
    } else {
        rgb = srgb_oetf(band_colour(band, lam));
    }
    out_px[gid.y * w + gid.x] = vec4<f32>(rgb, 1.0);
}
