#!/usr/bin/env python3
"""
Spectral raytracer for a round-brilliant diamond -- CPU reference.

This is the physics definition the GPU port is validated against
(gpu/diamond.py --compare). It is 40x slower; prefer the GPU one
for anything but a reference frame.

One randomly sampled wavelength per path (380-730 nm), weighted by the CIE 1931
colour matching functions. Cauchy dispersion, exact unpolarised Fresnel,
deterministic total internal reflection.

Requires: numpy, pillow
    pip install numpy pillow

Usage:
    python cpu/diamond.py                          # 512px, 48spp
    python cpu/diamond.py -w 800 -s 200 -o gem.png # slow, clean
    python cpu/diamond.py --fire 3 --azimuth 35    # exaggerated dispersion
    python cpu/diamond.py --preview                # 192px, 12spp, ~5s
    python cpu/diamond.py --ambient 0 --lights 1   # black void, max colour
    python cpu/diamond.py --rig showcase           # bright, and still sparkles
    python cpu/diamond.py --anim spin.mp4          # 60-frame rotation (ffmpeg)
    python cpu/diamond.py --top                    # straight-down table view

Lighting rigs (--rig) trade per-flash colour against how much of the stone is
lit. Measured at 384px/1024spp:

    rig         lit area   fiery area   mean saturation
    softbox      10.89%       2.72%          0.470
    showcase      9.46%       2.38%          0.500
    fire          3.95%       1.21%          0.544
    studio        1.78%       0.72%          0.633      <- default

softbox is bright and brilliant with little colour; fire is dark and contrasty
with the purest flashes; showcase mixes hard keys with broad fills and is the
best all-rounder. Hard keys are ~4x noisier per sample, so fire and showcase
want more spp than softbox for the same cleanliness.

Lighting note: --ambient trades colour saturation against readability. Measured
at 224px/24spp, 3 lights:

    ambient   mean saturation   fraction of frame lit
    1.00          0.454                 0.215
    0.50          0.482                 0.204
    0.25          0.496                 0.183      <- default
    0.12          0.498                 0.115
    0.05          0.600                 0.022

Saturation keeps climbing as the fill drops, but below ~0.25 the lit area falls
off a cliff and the stone stops reading as a solid object.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from cpu import animate, geometry, imaging, lighting, trace    # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-w", "--width", type=int, default=512)
    p.add_argument("-s", "--spp", type=int, default=48,
                   help="samples per pixel (one wavelength each)")
    p.add_argument("-b", "--bounces", type=int, default=8,
                   help="max internal reflections")
    p.add_argument("-o", "--out", default=os.path.join("rendered", "diamond.png"))
    p.add_argument("-e", "--exposure", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fire", type=float, default=1.0,
                   help="dispersion multiplier; 1.0 is physical diamond")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--elevation", type=float, default=26.0)
    p.add_argument("--distance", type=float, default=2.33)
    p.add_argument("--fov", type=float, default=34.0)
    p.add_argument("--ambient", type=float, default=0.25,
                   help="sky fill level; 0 = pure black void (max colour, "
                        "but the stone loses its form)")
    p.add_argument("--rig", choices=list(lighting.RIGS), default=lighting.DEFAULT_RIG,
                   help="lighting rig. softbox: broad sources, bright and "
                        "brilliant, little colour. studio: the original, "
                        "midway. fire: hard keys, dark and contrasty, most "
                        "saturated flashes. showcase: hard keys plus broad "
                        "fills, bright and still sparkling")
    p.add_argument("--lights", type=int, default=None, choices=range(1, 6),
                   help="how many of the rig's 5 lights to use "
                        "(default: the rig's own count)")
    p.add_argument("--chunk", type=int, default=24000,
                   help="rays per batch; lower it if you run out of memory")
    p.add_argument("--hdr", help="also save raw linear radiance as .npy")
    p.add_argument("--preview", action="store_true",
                   help="fast 192px / 12spp preview")
    p.add_argument("--top", action="store_true",
                   help="top-down view (looks straight down at the table)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                   help="number of worker processes to use (default: all available CPUs)")
    p.add_argument("--anim", metavar="OUT.mp4",
                   help="render a rotation animation to this .mp4 (needs ffmpeg)")
    p.add_argument("--anim-frames", type=int, default=60,
                   help="number of frames for --anim (default 60)")
    p.add_argument("--anim-step", type=float, default=0.1,
                   help="degrees the stone turns per frame (default 0.1). The "
                        "camera and lights stay put, so --azimuth is honoured "
                        "on every frame. A round brilliant repeats every 45 "
                        "deg, so 45/--anim-frames gives one full cycle")
    p.add_argument("--fps", type=int, default=30,
                   help="frame rate of the --anim .mp4 (default 30)")
    a = p.parse_args()
    if a.preview:
        a.width, a.spp = 192, 12
    if a.top:
        a.elevation = 90.0
    a.lights = lighting.apply_scene(a.rig, a.ambient, a.lights)

    if a.anim:
        animate.animate(a)
        return

    print(f"{len(geometry.TRI)} triangles | {a.width}x{a.width} | {a.spp} spp | "
          f"{a.bounces} internal reflections | fire x{a.fire:g} | "
          f"rig {a.rig} ({a.lights} lights) | ambient {a.ambient:g} | "
          f"jobs {a.jobs}")
    t0 = time.time()

    def show(done, total):
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done) if done else 0.0
        sys.stderr.write(f"\r  {done}/{total} spp  {elapsed:6.1f}s "
                         f"(eta {eta:6.1f}s)")
        sys.stderr.flush()

    try:
        img = trace.render_parallel(a.width, a.width, a.spp, a.bounces, a.seed, a.fire,
                              a.azimuth, a.elevation, a.distance, a.fov,
                              a.chunk, a.jobs, progress=show, rig=a.rig,
                              ambient=a.ambient, nlights=a.lights)
    except KeyboardInterrupt:
        sys.stderr.write("\nrender aborted by user\n")
        sys.exit(1)
    if a.hdr:
        imaging.save_npy(img, a.hdr)
    try:
        out = imaging.save_png((imaging.tonemap(img, a.exposure) * 255).astype(np.uint8), a.out)
    except ImportError:
        sys.exit("pillow not installed; rerun with --hdr to save raw .npy")
    print(f"wrote {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()