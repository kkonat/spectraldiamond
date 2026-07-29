# Spec: fixing washed-out spectral highlights in the diamond renderer

Hand-off notes for the agent maintaining the shader implementation. Nothing in
the light-transport code changes. Everything here is post-processing.

## Problem statement

Spectral dispersion is being computed correctly, but the display transform is
destroying the colour before it reaches the screen. Measured on a reference
render (brightest 0.5% of pixels, the ones carrying the fire):

| stage                          | mean saturation `(max-min)/max` |
| ------------------------------ | ------------------------------- |
| linear HDR, as rendered        | **0.664**                       |
| after per-channel ACES tonemap | **0.148**                       |

78% of the chroma is lost in tonemapping. Two separate mechanisms are
responsible, and both need fixing.

**Mechanism 1 — per-channel tone curves desaturate by construction.** Applying
any compressive curve independently to R, G and B lets the largest channel
saturate first while the others catch up. This is the intentional "filmic
path-to-white" look. A diamond's spectral flashes _are_ the highlights, so the
curve bleaches exactly the pixels that carry the hue.

**Mechanism 2 — naive gamut handling finishes the job.** Monochromatic
wavelengths lie on the spectral locus, far outside sRGB, so their linear RGB
has negative channels (measured minimum in the reference render: **−60.7**,
affecting 0.86% of pixels). Clamping per channel with `max(c, 0.0)` both
desaturates and twists hue.

## Critical precondition — check this first

The negative channel values must survive to the grading pass. Verify:

- The accumulation / resolve target is **float** (`RGBA16F` or `RGBA32F`), not
  a UNORM format. UNORM cannot represent negatives and silently clamps.
- No `max(color, 0.0)`, `clamp(color, 0.0, 1.0)` or saturate() appears anywhere
  between wavelength-to-RGB conversion and the grading pass.
- If temporal accumulation or denoising runs in between, it must not clamp
  either.

If any of these clamp, the out-of-gamut information is already gone and the
rest of this spec can only partially help. Fix this before anything else.

## Target pipeline

Order matters. Each stage depends on the previous one's range.

```
1. exposure          multiply linear HDR
2. bloom             DROPPED -- see the status note under stage 2
3. tone curve        on luminance only, chromaticity preserved
4. chroma boost      Oklab, AFTER the curve
   -> convert working space to sRGB here (see the follow-up section)
5. gamut fit         negatives first, then overflow via divide-by-max
6. levels            optional black/white/gamma
7. OETF              sRGB encode (skip if the swapchain is _SRGB)
```

Two ordering constraints worth stating explicitly, because getting them
backwards silently undoes the work:

- **Bloom must precede the tone curve.** Glare is a physical scattering effect
  in linear radiance. Blooming after compression spreads already-bleached white
  and adds nothing.
- **Chroma boost must follow the tone curve.** Boosting in HDR just pushes
  values that the curve will compress straight back down.

---

## Stage 3 — luminance-only tone curve

Replace the per-channel call. Keep the existing scalar ACES fit; only the
application changes.

```glsl
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

float aces(float x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 acesPerChannel(vec3 c) {   // the old path, kept for the blend
    return vec3(aces(c.r), aces(c.g), aces(c.b));
}

// strength: 1.0 = full chroma preservation, 0.0 = current behaviour
vec3 tonemapLuminance(vec3 c, float strength) {
    float L  = dot(c, LUMA);
    float Lt = aces(max(L, 0.0));
    vec3  lumPath = (L > 1e-6) ? c * (Lt / L) : vec3(0.0);
    return mix(acesPerChannel(max(c, 0.0)), lumPath, strength);
}
```

Expose `strength` as a uniform. It is the single most useful dial in the whole
change — art direction will want to sit somewhere below 1.0 on some shots.

Note the output of this stage **can exceed 1.0** in individual channels. That is
intended and is handled in stage 5. Do not clamp here.

---

## Stage 4 — Oklab chroma boost

Perceptually uniform and hue-stable. Do **not** substitute an HSV saturation
multiply — HSV skews hue on exactly the saturated colours we care about.

```glsl
vec3 linearToOklab(vec3 c) {
    vec3 lms = vec3(dot(c, vec3(0.4122214708, 0.5363325363, 0.0514459929)),
                    dot(c, vec3(0.2119034982, 0.6806995451, 0.1073969566)),
                    dot(c, vec3(0.0883024619, 0.2817188376, 0.6299787005)));
    vec3 r = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));   // sign-preserving
    return vec3(dot(r, vec3( 0.2104542553,  0.7936177850, -0.0040720468)),
                dot(r, vec3( 1.9779984951, -2.4285922050,  0.4505937099)),
                dot(r, vec3( 0.0259040371,  0.7827717662, -0.8086757660)));
}

vec3 oklabToLinear(vec3 lab) {
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return vec3( 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

// Vibrance rather than saturation: already-colourful pixels are boosted less,
// so the flashes intensify without the neutral body of the stone smearing.
vec3 boostChroma(vec3 c, float amount) {
    vec3  lab = linearToOklab(max(c, 0.0));
    float C   = length(lab.yz);
    float k   = 1.0 + (amount - 1.0) / (1.0 + 4.0 * C);
    return oklabToLinear(vec3(lab.x, lab.yz * k));
}
```

`sign()` returns 0 at exactly 0, which is correct here — the cube root of 0 is 0. Cost is roughly 20 ALU plus three `pow`, negligible against the trace.

---

## Stage 5 — gamut fit (the other half of the fix)

This is the stage that was quietly ruining things, and the reasoning matters
more than the code.

**A display cannot show a colour that is both bright and saturated.** Something
must be given up. Worked example, a bright spectral flash at linear
`(100.0, 5.0, 2.0)` with exposure 0.6:

| step                                 | result                               |
| ------------------------------------ | ------------------------------------ |
| after luminance tone curve           | `(4.003, 0.200, 0.080)`              |
| fit by desaturating toward luminance | `(1.000, 1.000, 1.000)` — pure white |
| fit by dividing by the max channel   | `(1.000, 0.050, 0.020)` — vivid red  |

Desaturating toward the luminance axis is the standard approach and it is
catastrophic here: at display luminance ≈ 1 there is no headroom left for
chroma, so the only in-gamut solution is white. Dividing by the max channel
preserves chromaticity exactly and pays in brightness instead. For gem fire
that is the trade we want.

```glsl
vec3 desaturateToWhite(vec3 c) {
    float L = clamp(dot(c, LUMA), 0.0, 1.0);
    vec3  d = c - vec3(L);
    vec3  hi = mix(vec3(1e9), (vec3(1.0) - L) / max(d, vec3( 1e-9)),
                   step(vec3(1e-9), d));
    vec3  lo = mix(vec3(1e9), (vec3(0.0) - L) / min(d, vec3(-1e-9)),
                   step(d, vec3(-1e-9)));
    float t = min(1.0, min(min(min(hi.x, hi.y), hi.z),
                           min(min(lo.x, lo.y), lo.z)));
    return clamp(vec3(L) + max(t, 0.0) * d, 0.0, 1.0);
}

// keepChroma: 1.0 = always sacrifice brightness (vivid, darker flashes)
//             0.0 = always sacrifice chroma  (bright, white flashes = old look)
vec3 fitGamut(vec3 c, float keepChroma) {
    // (a) clear negatives by desaturating the minimum necessary, constant hue
    float L  = max(dot(c, LUMA), 0.0);
    float mn = min(min(c.r, c.g), c.b);
    float t  = (mn < 0.0) ? clamp(L / max(L - mn, 1e-9), 0.0, 1.0) : 1.0;
    c = vec3(L) + t * (c - vec3(L));

    // (b) handle overflow with chromaticity intact
    vec3 chromaPath = c / max(max(max(c.r, c.g), c.b), 1.0);
    if (keepChroma >= 1.0) return clamp(chromaPath, 0.0, 1.0);
    return clamp(mix(desaturateToWhite(c), chromaPath, keepChroma), 0.0, 1.0);
}
```

Step (a) must run before step (b): dividing by the max while a channel is still
negative produces a hue inversion.

---

## Stage 2 — bloom

**Status: implemented, measured, then removed.** The prediction below did not
survive contact with this scene. It assumes above-threshold pixels carry a
modest share of frame energy; measured here they carry 92% of it (0.8% of
pixels, luminance p50 0.003 against p100 126) because a brilliant cut is nearly
all specular flash. At the spec's 0.35 that re-adds a third of the whole frame
as haze and dissolves the stone's form, and raising the threshold does not
help — 41% of the energy is still above 32. At high amounts the mip chain's
box-downsample also beats against the one-pixel flashes and crawls along facet
edges. Glare belongs in a compositor here, on an image the renderer has not
already softened. The rest of this section is kept as the record of what was
tried.

Expect this to be the largest _perceptual_ win, ahead of the colour maths. An
isolated one-pixel spectral flash reads as nothing on screen; bloomed into a
halo it reads as a coloured star. It is also physically motivated — corneal and
lens scatter do exactly this, which is why fire looks so vivid to the naked eye
and so flat in a render.

Standard mip-chain implementation, in linear HDR:

1. **Prefilter** into a half-res target: `max(hdr - threshold, 0.0)`. The
   subtraction naturally clears negatives, so the bloom chain itself does not
   need signed handling.
2. **Downsample** 5–6 times, each to half resolution, with a 13-tap or
   tent filter.
3. **Upsample** back up, additively accumulating with a 3×3 tent, so every
   octave contributes.
4. **Composite:** `final = hdr + amount * bloomAccum / levelCount`.

Do not use a single large-radius Gaussian. Multi-scale is what produces the
wide soft falloff plus tight core that reads as glare rather than as blur.

Threshold interacts with exposure — if exposure is applied before the
prefilter, keep threshold near 1.0; if after, scale it accordingly. Pick one
and document it, because this is a common source of "bloom looks different
after an exposure change" bugs.

---

## Uniforms and suggested defaults

| uniform           | default | range   | notes                                      |
| ----------------- | ------- | ------- | ------------------------------------------ |
| `exposure`        | 0.6     | 0.1–3   |                                            |
| `tonemapStrength` | 1.0     | 0–1     | 0 reproduces current output exactly        |
| `chromaBoost`     | 1.35    | 1.0–2.0 | above ~2 looks synthetic                   |
| `keepChroma`      | 0.85    | 0–1     | 1.0 is most vivid, slightly darker flashes |
| `workingSpace`    | rec2020 | —       | see the follow-up section                  |

`bloomAmount` and `bloomThreshold` are gone with stage 2.

`tonemapStrength = 0`, `chromaBoost = 1`, `keepChroma = 0`, `bloomAmount = 0`
should reproduce the current image bit-for-bit. Please verify that — it makes
the change reviewable and gives a safe rollback.

---

## Verification

Unit-testable without rendering:

1. **Neutral preservation.** Any input with `r == g == b` must remain neutral
   through the entire chain. If it does not, a matrix is transposed.
2. **Oklab round trip.** `oklabToLinear(linearToOklab(c)) ≈ c` to within 1e-6
   for random positive `c`. Test with values above 1.0 too, not just [0,1].
3. **Reference pixel.** Linear `(100.0, 5.0, 2.0)`, exposure 0.6,
   `tonemapStrength = 1`, `chromaBoost = 1`, `keepChroma = 1` must give
   `(1.000, 0.050, 0.020)` within 1e-3. With `keepChroma = 0` it must give
   `(1.000, 1.000, 1.000)`.
4. **Gamut closure.** `fitGamut` output must lie in [0,1] for all inputs
   including large negatives. Fuzz with values in [−100, 500].
5. **Rollback identity.** Defaults-to-zero settings match the current build.

---

## Explicitly out of scope — and why

**Do not attempt to fix this with curves, levels, or a saturation layer applied
to the LDR image.** Those operate after the damage, in display space. A pixel
already compressed to `(1,1,1)` contains no hue to recover, and pushing
saturation on what remains amplifies quantisation and ringing. Every fix above
is deliberately placed before the 8-bit encode. Levels are included as stage 6
for contrast shaping only, not as a colour fix.

**Do not swap the tone curve for a different one** (Reinhard, Uchimura, AgX) as
the primary fix. The curve is not the problem; applying it per channel is. Any
curve driven through `tonemapLuminance` will behave. AgX in particular is even
more aggressively desaturating in the highlights and will make this worse.

## Optional follow-up: wide-gamut working space — implemented

**Status: done, default `--working-space rec2020`.** Rec.2020 rather than
ACEScg because AP1 is a D60 space and would need a chromatic adaptation on top,
moving the white point of the studio lights; Rec.2020 shares sRGB's D65, so the
change is a pure primaries swap. The composed Rec.2020→sRGB matrix has its rows
renormalised to sum to 1: built from the published matrices they sum to 1 ± 3e-4,
because the sRGB matrix in use is the 5-digit rounded one and misses D65 on its
own, and that error tints every neutral and fails verification #1.

Measured, at 720px/2048spp on the mirror-floor frame:

| quantity                                     | sRGB   | Rec.2020 |
| -------------------------------------------- | ------ | -------- |
| locus chroma surviving stage 5(a)             | 55.5%  | 73.2%    |
| pixels still negative at stage 5              | 0.38%  | 0.16%    |
| mean saturation, brightest 0.5%               | 0.347  | 0.349    |
| saturation on the pixels that changed at all  | 0.948  | 0.963    |

The first two rows are large, the last two are not, and the reason is worth
recording: only 0.4% of pixels differ by more than 8/255 at all. Real pixels are
broadband mixtures of many wavelengths, and a mixture is already inside sRGB —
the wide gamut only helps the near-monochromatic minority. Those are exactly the
fire pixels, so the change is worth keeping, but it is a targeted fix and not a
visible lift across the frame. If the fire still reads short, the lever is
dispersion and lighting, not the colour pipeline.

The original note follows.

Beyond the scope of this change, but the logical next step if the fire still
reads short after the above. Convert XYZ to **Rec.2020 or ACEScg** primaries
instead of sRGB at the wavelength-to-RGB step, run the whole grade there, and
convert to sRGB with a gamut compression only at the very end. The spectral
locus fits far better inside Rec.2020, so far fewer pixels go negative in the
first place and stage 5(a) has much less work to do. This is what production
spectral renderers do. It is a one-matrix change at the source plus a new final
conversion, but it touches the renderer rather than just the post chain, so it
is worth doing as a separate reviewable commit.
