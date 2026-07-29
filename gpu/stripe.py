#!/usr/bin/env python3
"""
Spectrum swatch: 380-730 nm run through the grading chain, written to spectrum.png.

Every column is one wavelength, converted to linear sRGB with the same CIE fit
and white normalisation the renderer uses (imported from cpu/, so
there is one source of truth). The default single band answers the question
"what is the most saturated, brightest thing my display can do for this hue":
negatives are cleared by the minimum-necessary desaturation (spec stage 5a),
then the triple is divided by its largest channel, so max(r,g,b) == 1.0 exactly
and nothing clips. No tone curve -- a tone curve can only make it darker.

Under each band sits a reference strip: the same hue pushed to the display's
gamut corner (min channel 0, max channel 1), i.e. the most saturated thing the
monitor can emit there. Band and strip should be compared column for column.

--compare stacks the other looks underneath. --rainbow sweep swaps the
reference for a plain linear HSV rainbow over the whole hue circle.

Requires: numpy, pillow, wgpu.

The shader is gpu/stripe.wgsl.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                            # noqa: E402

from cpu import imaging, spectrum             # noqa: E402

SHADER = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "stripe.wgsl"), encoding="utf8").read()


F = np.float32

PARAMS = np.dtype([
    ("wnorm", "4f4"),   # xyz white normalisation of the CIE weights
    ("dims",  "4u4"),   # width, height, band count, band height in px
    ("cfg",   "4f4"),   # lam_min, lam_span, chroma_boost, keep_chroma
    ("cfg2",  "4f4"),   # exposure, separator px, reference strip px, mode
])


BAND_NAMES = [
    "max saturation + max lightness (chromaticity exact, max channel = 1.0)",
    "same, after Oklab vibrance boost, re-maxed",
    "shipping grade: luminance ACES -> boost -> fitGamut(keepChroma)",
    "old look: per-channel ACES",
]

RAINBOW_NAMES = {
    "corner": "monitor's most saturated colour at the same HSV hue (the ceiling)",
    "sweep":  "plain full-RGB rainbow, hue swept 300deg -> 0deg linearly",
    "off":    "",
}


def render(width, height, bands, sep, sub, mode, lam_min, lam_max, chroma_boost,
           keep_chroma, exposure, force_fallback=False):
    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(
        power_preference="low-power" if force_fallback else "high-performance")
    if adapter is None:
        sys.exit("no GPU adapter available")
    device = adapter.request_device_sync()

    band_h = height // bands
    height = band_h * bands                      # keep bands equal and exact

    p = np.zeros((), PARAMS)
    p["wnorm"][:3] = spectrum.WNORM
    p["dims"] = (width, height, bands, band_h)
    p["cfg"] = (lam_min, lam_max - lam_min, chroma_boost, keep_chroma)
    p["cfg2"] = (exposure, sep, sub, mode)

    ubo = device.create_buffer_with_data(
        data=p.tobytes(), usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    out = device.create_buffer(
        size=width * height * 4 * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    C = wgpu.ShaderStage.COMPUTE
    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": C,
         "buffer": {"type": wgpu.BufferBindingType.uniform}},
        {"binding": 1, "visibility": C,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        compute={"module": device.create_shader_module(code=SHADER),
                 "entry_point": "main"})
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": ubo, "offset": 0, "size": ubo.size}},
        {"binding": 1, "resource": {"buffer": out, "offset": 0, "size": out.size}},
    ])

    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((width + 7) // 8, (height + 7) // 8, 1)
    cp.end()
    device.queue.submit([enc.finish()])

    raw = np.frombuffer(device.queue.read_buffer(out), F)
    info = adapter.info
    return (raw.reshape(height, width, 4)[:, :, :3],
            f"{info['device']} ({info['backend_type']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=os.path.join("rendered", "spectrum.png"))
    ap.add_argument("-W", "--width", type=int, default=1400)
    ap.add_argument("-H", "--height", type=int, default=200,
                    help="height of one band")
    ap.add_argument("--compare", action="store_true",
                    help="stack the boosted, graded and old-look bands underneath")
    ap.add_argument("--range", type=float, nargs=2, metavar=("MIN", "MAX"),
                    default=(spectrum.LMIN, spectrum.LMAX),
                    help="wavelength range in nm (default: the renderer's band)")
    ap.add_argument("--chroma-boost", type=float, default=1.35)
    ap.add_argument("--keep-chroma", type=float, default=0.85)
    ap.add_argument("--exposure", type=float, default=1.0,
                    help="exposure for the tonemapped comparison bands only")
    ap.add_argument("--sep", type=int, default=3, help="separator height in px")
    ap.add_argument("--rainbow", choices=("corner", "sweep", "off"),
                    default="corner",
                    help="reference strip under each band: 'corner' = the "
                         "monitor's most saturated colour at the same hue, "
                         "'sweep' = a plain linear HSV rainbow, 'off' = none")
    ap.add_argument("--rainbow-height", type=int, default=0,
                    help="reference strip height in px (default: a third of the band)")
    ap.add_argument("--cpu", action="store_true", help="force the fallback adapter")
    a = ap.parse_args()

    lam_min, lam_max = a.range
    bands = 4 if a.compare else 1
    sub = 0 if a.rainbow == "off" else (a.rainbow_height or a.height // 3)
    mode = 1 if a.rainbow == "sweep" else 0
    band_h = a.height + sub

    img, gpu = render(a.width, band_h * bands, bands, a.sep, sub, mode,
                      lam_min, lam_max, a.chroma_boost, a.keep_chroma,
                      a.exposure, a.cpu)

    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    imaging.save_png(u8, a.out)

    print(f"{gpu}\n{a.out}  {u8.shape[1]}x{u8.shape[0]}  "
          f"{lam_min:.0f}-{lam_max:.0f} nm left to right")
    for i in range(bands):
        print(f"  band {i}: {BAND_NAMES[i]}")
    if sub:
        print(f"  strip under each band: {RAINBOW_NAMES[a.rainbow]}")

    # How much of the display's reach each band actually uses. Saturation is
    # the spec's (max-min)/max on the 8-bit triple; the delta is the largest
    # channel difference between a band and the strip below it.
    if sub:
        def sat(a_):
            mx = a_.max(-1).astype(np.int16)
            mn = a_.min(-1).astype(np.int16)
            return (mx - mn) / np.maximum(mx, 1)

        print("\n  band   mean sat   ref sat   max channel delta vs ref")
        for i in range(bands):
            top = i * band_h + a.sep
            main = u8[top:i * band_h + a.height, :, :]
            rstrip = u8[i * band_h + a.height + a.sep:(i + 1) * band_h, :, :]
            d = np.abs(main.mean(0).astype(np.int16)
                       - rstrip.mean(0).astype(np.int16)).max()
            print(f"  {i:4d}     {sat(main).mean():.3f}      "
                  f"{sat(rstrip).mean():.3f}     {d:3d}/255")

    # Spot values off the top band and its reference strip, so the headroom can
    # be read as numbers too. Saturation here is the spec's (max-min)/max.
    def sat(px):
        mx, mn = int(px.max()), int(px.min())
        return (mx - mn) / max(mx, 1)

    head = "\n  nm     model sRGB      hex     "
    print(head + ("  monitor max    sat model / max" if sub else ""))
    for lam in (400, 440, 470, 490, 510, 540, 570, 590, 610, 640, 680, 720):
        if not lam_min <= lam <= lam_max:
            continue
        x = int((lam - lam_min) / (lam_max - lam_min) * a.width)
        x = min(max(x, 0), a.width - 1)
        r, g, b = u8[a.height // 2, x]
        line = f"  {lam:4d}   {r:3d} {g:3d} {b:3d}      #{r:02x}{g:02x}{b:02x}"
        if sub:
            m = u8[band_h - sub // 2, x]
            line += (f"    #{m[0]:02x}{m[1]:02x}{m[2]:02x}"
                     f"        {sat(u8[a.height // 2, x]):.2f} / {sat(m):.2f}")
        print(line)

    if lam_min < 400 or lam_max > 700:
        print("\n  note: outside ~400-700 nm the Wyman CIE fit's tails carry"
              "\n  essentially no radiance and its hue there is meaningless."
              "\n  Normalising to max lightness amplifies that to full brightness"
              "\n  (the violet end reads blue, the far red end reads yellow-green)."
              "\n  In the render those wavelengths contribute nothing; use"
              "\n  --range 400 700 for a swatch of the part that matters.")


if __name__ == "__main__":
    main()
