const PI: f32 = 3.141592653589793;
const LMIN: f32 = 380.0;
const LSPAN: f32 = 350.0;
const NOHIT: f32 = 1e30;
const DIM_BG: f32 = 0.16;   // reference model's background-only dimming

struct Params {
    eye:    vec4<f32>,
    fwd:    vec4<f32>,
    right:  vec4<f32>,
    up:     vec4<f32>,
    wnorm:  vec4<f32>,
    bs_c:   vec4<f32>,
    cauchy: vec4<f32>,
    misc:    vec4<f32>,
    floor_p: vec4<f32>,
    floor_s: vec4<f32>,
    floor_a: vec4<f32>,
    floor_b: vec4<f32>,
    x2ws:   array<vec4<f32>, 3>,
    sky:    array<vec4<f32>, 2>,
    cfg:    vec4<u32>,
    cfg2:   vec4<u32>,
    cfg3:   vec4<u32>,
    lights: array<vec4<f32>, 5>,
    lint:   array<vec4<f32>, 5>,
};

@group(0) @binding(0) var<uniform> P: Params;
// 4 vec4 per triangle: v0, e1, e2, normal.
@group(0) @binding(1) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;

// ------------------------------------------------------------------- random
fn pcg(state: ptr<function, u32>) -> f32 {
    var s = *state * 747796405u + 2891336453u;
    *state = s;
    var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w) * 2.3283064365386963e-10;
}

fn seed_of(pix: u32, samp: u32, salt: u32) -> u32 {
    var h = pix * 747796405u + samp * 2891336453u + salt * 2654435761u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    return h ^ (h >> 16u);
}

// ------------------------------------------------------ spectral <-> colour
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
    // Primaries come from the host: sRGB or a wide-gamut working space.
    let xyz = vec3<f32>(X, Y, Z);
    let rgb = vec3<f32>(dot(P.x2ws[0].xyz, xyz),
                        dot(P.x2ws[1].xyz, xyz),
                        dot(P.x2ws[2].xyz, xyz));
    return rgb / P.wnorm.xyz;
}

// Three smooth windows forming a partition of unity: a neutral RGB radiance
// samples to a flat spectrum exactly.
fn rgb_to_spectral(c: vec3<f32>, lam: f32) -> f32 {
    let t = clamp((lam - LMIN) / LSPAN, 0.0, 1.0);
    let b = 0.5 * (1.0 + cos(PI * clamp((t - 0.15) / 0.25, 0.0, 1.0)));
    let r = 0.5 * (1.0 - cos(PI * clamp((t - 0.55) / 0.25, 0.0, 1.0)));
    let g = 1.0 - b - r;
    return c.x * r + c.y * g + c.z * b;
}

// -------------------------------------------------------------- environment
fn sky_only(d: vec3<f32>) -> vec3<f32> {
    let y = d.y;
    var sky: f32;
    if (y > 0.0) {
        sky = 0.28 + 0.42 * clamp(y, 0.0, 1.0);
    } else {
        sky = 0.16 * exp(4.0 * clamp(y, -1.0, 0.0));
    }
    var col = sky * P.sky[0].xyz;
    if (y < -0.05) {
        col = col + P.sky[1].xyz;
    }
    return col * P.misc.x;              // ambient scales the fill only
}

// Sky plus the studio lights, each lobe optionally widened by `extra` and
// dimmed to conserve its energy. extra = 0 is the exact light model; a glossy
// floor uses extra > 0, which is both the right look and a large variance
// reduction (the lights are only ~0.05% of the sphere at intensity 300+).
fn env_gloss(d: vec3<f32>, extra: f32) -> vec3<f32> {
    var col = sky_only(d);
    let nl = P.cfg2.w;
    for (var i = 0u; i < nl; i = i + 1u) {
        let L = P.lights[i];
        let w = L.w + extra;
        let s = clamp(dot(d, L.xyz), -1.0, 1.0);
        col = col + P.lint[i].x * (L.w / w) * exp(-(1.0 - s) / w);
    }
    return col;
}

fn env(d: vec3<f32>, primary: bool) -> vec3<f32> {
    if (primary) {
        return sky_only(d) * DIM_BG;
    }
    return env_gloss(d, 0.0);
}

// ------------------------------------------------------------------ shading
// Returns (reflectance, tir_flag).
fn fresnel(cosi: f32, eta: f32) -> vec2<f32> {
    let sin2 = eta * eta * (1.0 - cosi * cosi);
    if (sin2 >= 1.0) {
        return vec2<f32>(1.0, 1.0);
    }
    let cost = sqrt(clamp(1.0 - sin2, 0.0, 1.0));
    let rs = (eta * cosi - cost) / max(eta * cosi + cost, 1e-7);
    let rp = (cosi - eta * cost) / max(cosi + eta * cost, 1e-7);
    return vec2<f32>(clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0), 0.0);
}

fn refract_dir(I: vec3<f32>, N: vec3<f32>, eta: f32, cosi: f32) -> vec3<f32> {
    let k = max(1.0 - eta * eta * (1.0 - cosi * cosi), 0.0);
    return eta * I + (eta * cosi - sqrt(k)) * N;
}

fn schlick(cosi: f32, f0: f32) -> f32 {
    let m = clamp(1.0 - cosi, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (1.0 - f0) * m2 * m2 * m;
}

// ----------------------------------------------------------------- geometry
struct Hit { t: f32, idx: u32 };

// Moller-Trumbore over every triangle, bounding-sphere prefiltered. The whole
// mesh is small enough that registers beat any acceleration structure.
fn hit_gem(o: vec3<f32>, d: vec3<f32>, tmin: f32) -> Hit {
    var h: Hit;
    h.t = NOHIT;
    h.idx = 0u;
    let oc = o - P.bs_c.xyz;
    let b = dot(oc, d);
    let c = dot(oc, oc) - P.bs_c.w * P.bs_c.w;
    let disc = b * b - c;
    if (disc <= 0.0 || (-b + sqrt(disc)) <= tmin) {
        return h;
    }
    let n = P.cfg.z;
    for (var i = 0u; i < n; i = i + 1u) {
        let j = 4u * i;
        let e1 = tris[j + 1u].xyz;
        let e2 = tris[j + 2u].xyz;
        let pv = cross(d, e2);
        let det = dot(pv, e1);
        if (abs(det) <= 1e-9) { continue; }
        let inv = 1.0 / det;
        let tv = o - tris[j].xyz;
        let u = dot(tv, pv) * inv;
        if (u < -1e-6) { continue; }
        let qv = cross(tv, e1);
        let v = dot(d, qv) * inv;
        if (v < -1e-6 || u + v > 1.0 + 1e-6) { continue; }
        let t = dot(qv, e2) * inv;
        if (t > tmin && t < h.t) {
            h.t = t;
            h.idx = i;
        }
    }
    return h;
}

// -------------------------------------------------------------------- floor
// A horizontal plane at floor_p.x. Mode 0 disables it entirely, so every
// branch below is inert and the render matches the no-floor result exactly.
fn hit_floor(o: vec3<f32>, d: vec3<f32>, tmin: f32) -> f32 {
    if (P.floor_p.w < 0.5 || abs(d.y) < 1e-6) {
        return NOHIT;
    }
    let t = (P.floor_p.x - o.y) / d.y;
    if (t <= tmin) {
        return NOHIT;
    }
    return t;
}

// Antialiased square wave, period 1, values 0/1. `w` is the ray footprint in
// periods; as it grows the wave collapses to flat grey instead of aliasing.
fn sq(x: f32, w: f32) -> f32 {
    let e = clamp(w, 0.0015, 0.5);
    let f = fract(x);
    let s = smoothstep(0.0, e, f) - smoothstep(0.5, 0.5 + e, f);
    return mix(s, 0.5, smoothstep(0.2, 0.5, w));
}

// Antialiased thin line at every integer, for the grid mode.
fn rule(x: f32, w: f32) -> f32 {
    let f = abs(fract(x) - 0.5) * 2.0;
    return (1.0 - smoothstep(0.0, clamp(w, 0.004, 1.0) * 2.0 + 0.03, f))
           * (1.0 - smoothstep(0.15, 0.45, w));
}

fn floor_albedo(q: vec2<f32>, foot: f32) -> vec3<f32> {
    let mode = u32(P.floor_p.w + 0.5);
    let p = q / P.floor_p.y;            // pattern-space coordinate
    let w = foot / P.floor_p.y;         // footprint in pattern space
    var t = 0.0;
    if (mode == 2u) {                   // sweep: smooth radial falloff
        t = smoothstep(0.0, 1.0, length(p) * 0.5);
    } else if (mode == 3u) {            // checker (xor of two square waves)
        t = abs(sq(p.x * 0.5, w * 0.5) - sq(p.y * 0.5, w * 0.5));
    } else if (mode == 4u) {            // stripes
        t = sq(p.x * 0.5, w * 0.5);
    } else if (mode == 5u) {            // concentric rings about the stone
        t = sq(length(p) * 0.5, w * 0.5);
    } else if (mode == 6u) {            // thin grid lines
        t = max(rule(p.x * 0.5, w * 0.5), rule(p.y * 0.5, w * 0.5));
    }                                   // mode 1 (mirror) stays flat
    return mix(P.floor_a.xyz, P.floor_b.xyz, t);
}

// Fraction of the studio lights' energy reaching hp, one stochastic sample per
// light. The sources are discs of finite angular size, so sampling them gives a
// real penumbra for free: tight under the culet where the stone is close, wide
// out at the edge of the shadow, and wider still for the broad fills.
//
// The sampling is exact rather than approximate. Solid angle factors as
// dOmega = d(1 - cos t) dphi, and the light profile is exp(-(1 - cos t)/w), so
// drawing (1 - cos t) from an exponential of mean w importance-samples the
// source perfectly -- no rejection loop, no cone approximation, two randoms.
//
// Weights are each light's irradiance on an up-facing surface, intensity * w
// for the solid angle it covers times the cosine, so a dim broad fill cannot
// darken the floor as much as the bright hard key does.
fn light_vis(hp: vec3<f32>, st: ptr<function, u32>) -> f32 {
    let nl = P.cfg2.w;
    var lit = 0.0;
    var tot = 0.0;
    for (var i = 0u; i < nl; i = i + 1u) {
        let n = P.lights[i].xyz;
        let w = P.lights[i].w;
        let wgt = P.lint[i].x * w * max(n.y, 0.0);
        if (wgt <= 0.0) { continue; }
        tot = tot + wgt;
        let x = -w * log(max(pcg(st), 1e-9));          // 1 - cos(theta)
        let ct = 1.0 - x;
        let sn = sqrt(max(1.0 - ct * ct, 0.0));
        let phi = 2.0 * PI * pcg(st);
        let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
                       abs(n.x) > 0.5);
        let t1 = normalize(cross(a, n));
        let t2 = cross(n, t1);
        let dir = normalize(ct * n + sn * (cos(phi) * t1 + sin(phi) * t2));
        if (hit_gem(hp, dir, 1e-3).t >= NOHIT) { lit = lit + wgt; }
    }
    if (tot <= 0.0) { return 1.0; }
    return lit / tot;
}

// Sky occlusion by the stone, from the solid angle its bounding sphere covers.
// Without it a backdrop-lit floor makes the gem look like it is floating.
fn floor_ao(hp: vec3<f32>) -> f32 {
    let v = hp - P.bs_c.xyz;
    let occ = clamp(P.bs_c.w * P.bs_c.w / (2.0 * max(dot(v, v), 1e-6)), 0.0, 1.0);
    return 1.0 - P.floor_s.w * occ;
}

// The backdrop is treated as emissive (a lightbox / backlit sweep), which is
// what makes the pattern read clearly through the stone. `spec` folds in the
// mirror term analytically; the camera path instead samples it stochastically.
fn floor_col(hp: vec3<f32>, d: vec3<f32>, t: f32, spec: bool, dim: f32,
             st: ptr<function, u32>) -> vec3<f32> {
    let foot = P.misc.z * t / max(abs(d.y), 0.03);
    // misc.w is the share of the backdrop's illumination taken to come from the
    // studio lights, and so the share the stone can block. At 0 the backdrop is
    // purely emissive and nothing below runs, which is the original behaviour
    // bit for bit. Note that outside the shadow light_vis is 1 and this term is
    // exactly 1 too, so enabling shadows only ever darkens occluded ground --
    // it never re-exposes the frame.
    var shade = 1.0;
    if (P.misc.w > 0.0) {
        shade = 1.0 - P.misc.w * (1.0 - light_vis(hp, st));
    }
    var col = floor_albedo(hp.xz, foot) * P.floor_s.x * floor_ao(hp) * shade;
    if (spec) {
        col = col + schlick(abs(d.y), P.floor_s.y)
                  * env_gloss(vec3<f32>(d.x, -d.y, d.z), P.floor_s.z);
    }
    // Dissolve into the environment near the horizon: no hard edge, and no
    // infinitely-aliasing pattern at grazing angles.
    let f = smoothstep(P.floor_p.z * 0.35, P.floor_p.z, length(hp.xz));
    return mix(col, sky_only(d) * dim, f);
}

// Environment lookup for a ray leaving the gem: the floor if it is in the way.
fn escape(o: vec3<f32>, d: vec3<f32>, st: ptr<function, u32>) -> vec3<f32> {
    let tf = hit_floor(o, d, 1e-4);
    if (tf < NOHIT) {
        return floor_col(o + tf * d, d, tf, true, 1.0, st);
    }
    return env_gloss(d, 0.0);
}

// ------------------------------------------------------------------- kernel
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let pix = (wid.y * P.cfg3.x + wid.x) * 64u + lid.x;
    if (pix >= P.cfg3.y) { return; }

    let W = P.cfg.x;
    let fw = f32(W);
    let fh = f32(P.cfg.y);
    let pxi = f32(pix % W);
    let pyi = f32(pix / W);
    let scale = P.misc.y;
    let nsamp = P.cfg2.z;
    let bounces = P.cfg.w;

    var total = vec3<f32>(0.0);

    for (var s = 0u; s < nsamp; s = s + 1u) {
        var rs = seed_of(pix, P.cfg2.y + s, P.cfg2.x);

        let sx = (2.0 * (pxi + pcg(&rs)) / fw - 1.0) * scale;
        let sy = (1.0 - 2.0 * (pyi + pcg(&rs)) / fh) * scale;
        var d = normalize(P.fwd.xyz + sx * P.right.xyz + sy * P.up.xyz);
        var o = P.eye.xyz;

        let lam = LMIN + LSPAN * pcg(&rs);
        let um = lam * 1e-3;
        let ng = P.cauchy.x + P.cauchy.y / (um * um);
        let wg = spectral_weight(lam);

        // ---- camera ray -> gem surface, with at most one specular bounce off
        //      the floor on the way, which is what puts the stone's own
        //      reflection into the surface below it. Reflecting with
        //      probability = Fresnel keeps the path weight at 1, exactly as
        //      the gem's own reflect/refract split does.
        var h: Hit;
        h.t = NOHIT;
        h.idx = 0u;
        var rad = vec3<f32>(0.0);
        var escaped = true;
        var bounced = false;
        for (var fb = 0u; fb < 2u; fb = fb + 1u) {
            h = hit_gem(o, d, 1e-4);
            let tf = hit_floor(o, d, 1e-4);
            if (tf < h.t) {
                let hp = o + tf * d;
                if (fb == 0u && pcg(&rs) < schlick(abs(d.y), P.floor_s.y)) {
                    o = hp;
                    d = vec3<f32>(d.x, -d.y, d.z);
                    bounced = true;
                    continue;
                }
                rad = floor_col(hp, d, tf, false, DIM_BG, &rs);
                break;
            }
            if (h.t >= NOHIT) {
                // Background, whether seen directly or via the floor, carries
                // the same dimming -- otherwise the floor reflects a sky 6x
                // brighter than the sky itself and the horizon becomes a seam.
                // The stone's reflection is not background, so it stays full
                // brightness: that is the whole point of the mirror floor.
                if (bounced) {
                    rad = env_gloss(d, P.floor_s.z) * DIM_BG;
                } else {
                    rad = sky_only(d) * DIM_BG;     // == env(d, true)
                }
                break;
            }
            escaped = false;
            break;
        }
        if (escaped) {
            total = total + wg * rgb_to_spectral(rad, lam);
            continue;
        }
        var p = o + h.t * d;
        var nn = tris[4u * h.idx + 3u].xyz;
        var ci = -dot(nn, d);
        if (ci < 0.0) { nn = -nn; ci = -ci; }

        let eta = 1.0 / ng;
        let fr = fresnel(ci, eta);
        if (pcg(&rs) < fr.x) {                      // reflected off the surface
            total = total + wg * rgb_to_spectral(escape(p, d + 2.0 * ci * nn, &rs), lam);
            continue;
        }
        d = normalize(refract_dir(d, nn, eta, ci));
        o = p;

        // ---- internal transport
        for (var bb = 0u; bb <= bounces; bb = bb + 1u) {
            h = hit_gem(o, d, 2e-4);
            if (h.t >= NOHIT) { break; }
            p = o + h.t * d;
            nn = tris[4u * h.idx + 3u].xyz;
            if (dot(nn, d) < 0.0) { nn = -nn; }     // orient along travel
            ci = abs(dot(nn, d));
            let f2 = fresnel(ci, ng);
            let u = pcg(&rs);
            if (f2.y == 0.0 && u >= f2.x) {         // refracts out and escapes
                let od = normalize(refract_dir(d, -nn, ng, ci));
                total = total + wg * rgb_to_spectral(escape(p, od, &rs), lam);
                break;
            }
            d = d - 2.0 * dot(nn, d) * nn;          // total internal reflection
            o = p;
        }
    }

    let base = 3u * pix;
    acc[base]      = acc[base]      + total.x;
    acc[base + 1u] = acc[base + 1u] + total.y;
    acc[base + 2u] = acc[base + 2u] + total.z;
}
