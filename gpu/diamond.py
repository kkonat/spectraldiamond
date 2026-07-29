#!/usr/bin/env python3
"""
GPU spectral raytracer for a round-brilliant diamond -- WGSL compute shader.

The kernel itself is gpu/kernel.wgsl; this file is only the command line.

Physically identical to cpu/diamond.py (same geometry, same Cauchy
dispersion, same unpolarised Fresnel, same CIE weighting), but the whole path
loop lives in one compute kernel: **one GPU thread per ray**.

Why a compute shader and not CuPy / torch?
    The numpy version's hit_gem() materialises `rays x 110` intermediate arrays
    (pv, det, u, v, t ...) on every bounce -- at the default 24000-ray chunk
    that is ~10 MB each, tens of MB of memory traffic per call. That structure
    is fundamentally memory-bound, and an array-backend port to
    CuPy or torch would faithfully inherit it. A compute shader instead keeps
    the 110 triangles in registers, runs all bounces inside the kernel, and
    never writes an intermediate to memory. Ray compaction is free: a lane
    whose path has terminated simply exits the loop.

Backend is wgpu (Vulkan / DX12 / Metal), so no CUDA toolkit is required and it
is not vendor-locked.

Requires: numpy, pillow, wgpu
    pip install numpy pillow wgpu

Usage:
    python gpu/diamond.py                          # 512px, 48spp
    python gpu/diamond.py -w 1024 -s 512           # still seconds
    python gpu/diamond.py --preview                # 192px, 12spp
    python gpu/diamond.py --top --fire 3
    python gpu/diamond.py --floor                  # black glossy acrylic
    python gpu/diamond.py --floor rings --top      # refraction showcase
    python gpu/diamond.py --rig showcase --floor   # product shot
    python gpu/diamond.py --rig fire -s 2048       # max colour, needs spp
    python gpu/diamond.py --anim spin.mp4          # needs ffmpeg
    python gpu/diamond.py --compare                # CPU/GPU agreement
    python gpu/diamond.py --list-adapters

--floor puts the stone on a studio surface, which the CPU version does not have.
It is a real plane, not an environment trick: a camera ray reflects off it with
probability equal to its Fresnel term and can then enter the gem, so the stone's
own reflection appears in the floor. Rays leaving the gem also land on it, which
is what makes the patterned modes worth using -- refraction shears the pattern
into the dispersion fan.

    mirror    black glossy acrylic. What gem and jewellery photography is
              actually shot on, and the default for a bare --floor: the
              backdrop is nearly unlit, so the reflection is the whole image.
    sweep     graduated seamless backdrop, bright under the stone.
    checker   classic checkerboard; the strongest read on refractive distortion.
    stripes   parallel bands, sheared hard by the pavilion.
    rings     concentric rings centred on the stone. Best with --top.
    grid      thin bright lines on black; least competition with the gem.

Pattern edges are widened with distance from the ray footprint, so the floor
dissolves into flat grey at the horizon instead of aliasing, then fades into the
sky over --floor-fade. --floor-bright 0 leaves only the reflection.

The kernel is dispatched in passes of a few samples each. Pass size is tuned at
runtime so a single dispatch stays well under the Windows GPU watchdog (TDR)
limit of ~2 s, which also gives usable progress reporting.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                              # noqa: E402

from cpu import imaging, lighting                               # noqa: E402
from gpu import animate, colour, grading, renderer, selftest    # noqa: E402
from gpu.progress import _progress, shadow_note                 # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-w", "--width", type=int, default=512)
    p.add_argument("-s", "--spp", type=int, default=128,
                   help="samples per pixel (one wavelength each)")
    p.add_argument("-b", "--bounces", type=int, default=8,
                   help="max internal reflections")
    p.add_argument("-o", "--out", default=os.path.join("rendered", "diamond_gpu.png"))
    p.add_argument("-e", "--exposure", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fire", type=float, default=1.0,
                   help="dispersion multiplier; 1.0 is physical diamond")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--elevation", type=float, default=26.0)
    p.add_argument("--distance", type=float, default=2.33)
    p.add_argument("--fov", type=float, default=34.0)
    p.add_argument("--ambient", type=float, default=0.08,
                   help="sky fill level; 0 = pure black void")
    p.add_argument("--rig", choices=list(lighting.RIGS), default=lighting.DEFAULT_RIG,
                   help="lighting rig. softbox: broad sources, bright and "
                        "brilliant, little colour. studio: the original, "
                        "midway. fire: hard keys, dark and contrasty, most "
                        "saturated flashes. showcase: hard keys plus broad "
                        "fills, bright and still sparkling. Hard keys need "
                        "more spp -- they are ~4x noisier per sample")
    p.add_argument("--lights", type=int, default=None, choices=range(1, 6),
                   help="how many of the rig's 5 lights to use "
                        "(default: the rig's own count)")
    p.add_argument("--hdr", help="also save raw linear radiance as .npy")
    p.add_argument("--preview", action="store_true",
                   help="fast 192px / 12spp preview")
    p.add_argument("--top", action="store_true",
                   help="top-down view (looks straight down at the table)")
    g = p.add_argument_group(
        "display grading",
        "post-processing chain from spectral_grading_spec.md, applied in spec "
        "order: exposure, luminance tone curve, Oklab chroma boost, convert to "
        "sRGB, gamut fit, levels, OETF. The spec's bloom stage is not "
        "implemented -- see the Grade docstring for why")
    g.add_argument("--working-space", choices=list(colour.SPACES), default="rec2020",
                   help="primaries the render accumulates and grades in. "
                        "rec2020 covers far more of the spectral locus, so "
                        "fewer flashes are desaturated to reach the display; "
                        "srgb is the pre-follow-up behaviour")
    g.add_argument("--tonemap-strength", type=float, default=1.0,
                   help="1 = curve on luminance only, chromaticity preserved; "
                        "0 = old per-channel ACES")
    g.add_argument("--chroma-boost", type=float, default=1.35,
                   help="Oklab vibrance after the curve; above ~2 looks synthetic")
    g.add_argument("--keep-chroma", type=float, default=0.85,
                   help="gamut fit: 1 = sacrifice brightness (vivid flashes), "
                        "0 = sacrifice chroma (white flashes)")
    g.add_argument("--levels", type=float, nargs=3, default=(0.0, 1.0, 1.0),
                   metavar=("BLACK", "WHITE", "GAMMA"),
                   help="optional contrast shaping before the OETF")
    g.add_argument("--legacy-grade", action="store_true",
                   help="restore the pre-spec look exactly (rollback switch)")
    g.add_argument("--selftest", action="store_true",
                   help="run the spec's verification suite and exit")
    p.add_argument("--floor", nargs="?", const="mirror", default="none",
                   choices=list(renderer.FLOORS),
                   help="put the stone on a studio floor. Bare --floor gives "
                        "'mirror', the black glossy acrylic gems are normally "
                        "photographed on. 'sweep' is a graduated backdrop; "
                        "'checker'/'stripes'/'rings'/'grid' are patterns meant "
                        "to be seen through the stone, where refraction shears "
                        "them into the dispersion fan")
    p.add_argument("--floor-y", type=float, default=-0.43,
                   help="floor height; the default is the culet, so the stone "
                        "just touches it")
    p.add_argument("--floor-scale", type=float, default=0.0,
                   help="pattern feature size in gem diameters; 0 = per-mode default")
    p.add_argument("--floor-bright", type=float, default=-1.0,
                   help="backdrop emission; 0 = unlit, so only the reflection shows")
    p.add_argument("--floor-gloss", type=float, default=-1.0,
                   help="floor reflectance at normal incidence (Schlick F0); "
                        "raise it for a stronger mirror image of the stone")
    p.add_argument("--floor-fade", type=float, default=6.0,
                   help="radius over which the floor dissolves into the sky")
    p.add_argument("--floor-ao", type=float, default=0.55,
                   help="contact shadow strength under the stone; 0 disables")
    p.add_argument("--shadow", type=float, nargs="?", const=0.7, default=0.0,
                   metavar="STRENGTH",
                   help="cast a shadow of the stone on the floor by tracing "
                        "one sampled shadow ray per light, giving a real "
                        "penumbra. The value is how much of the backdrop's "
                        "light is taken to come from the studio lights and can "
                        "therefore be blocked: bare --shadow gives 0.7, 1.0 is "
                        "black under the stone. Needs --floor")
    p.add_argument("--pass-spp", type=int, default=0,
                   help="samples per dispatch; 0 = auto-tune to ~0.35s/pass")
    p.add_argument("--chunk", type=int, default=24000,
                   help="CPU-side ray batch, only used by --compare")
    p.add_argument("--low-power", action="store_true",
                   help="prefer the integrated GPU")
    p.add_argument("--list-adapters", action="store_true",
                   help="show available GPU adapters and exit")
    p.add_argument("--compare", action="store_true",
                   help="render small on both backends and report agreement")
    p.add_argument("--anim", metavar="OUT.mp4",
                   help="render a rotation animation to this .mp4 (needs ffmpeg)")
    p.add_argument("--anim-frames", type=int, default=60)
    p.add_argument("--anim-step", type=float, default=0.1,
                   help="degrees the stone turns per frame. The camera and "
                        "lights stay put, so --azimuth is honoured on every "
                        "frame. A round brilliant repeats every 45 deg, so "
                        "45/--anim-frames gives one full cycle")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()

    if a.list_adapters:
        renderer.list_adapters()
        return
    if a.preview:
        a.width, a.spp = 192, 12
    if a.top:
        a.elevation = 90.0

    a.floor_spec = renderer.FloorSpec(a.floor, a.floor_y, a.floor_scale, a.floor_bright,
                             a.floor_gloss, a.floor_fade, a.floor_ao)
    # Selects lighting.LIGHTS, which the renderer packs into the uniform block, and
    # resolves --lights against the rig's own light count.
    a.lights = lighting.apply_scene(a.rig, nlights=a.lights)
    if a.legacy_grade:
        a.tonemap_strength, a.chroma_boost, a.keep_chroma = 0.0, 1.0, 0.0
        a.levels, a.working_space = (0.0, 1.0, 1.0), "srgb"
    a.space = colour.working_space(a.working_space)
    a.grade = grading.Grade(a.exposure, a.tonemap_strength, a.chroma_boost,
                    a.keep_chroma, a.levels, a.space)
    gpu = renderer.GPURenderer(force_fallback=a.low_power)

    if a.selftest:
        sys.exit(selftest.selftest(gpu))
    if a.compare:
        selftest.compare(gpu, a)
        return
    if a.anim:
        animate.animate(gpu, a)
        return

    print(f"{gpu.name} | {gpu.ntri} triangles | {a.width}x{a.width} | "
          f"{a.spp} spp | {a.bounces} internal reflections | "
          f"fire x{a.fire:g} | rig {a.rig} ({a.lights} lights) | "
          f"ambient {a.ambient:g} | "
          f"{a.floor_spec}{shadow_note(a)} | {a.grade}")
    t0 = time.time()
    try:
        img = gpu.render(a.width, a.width, a.spp, a.bounces, a.seed, a.fire,
                         a.azimuth, a.elevation, a.distance, a.fov,
                         a.ambient, a.lights, a.pass_spp, progress=_progress(),
                         floor=a.floor_spec, space=a.space, shadow=a.shadow)
    except KeyboardInterrupt:
        sys.stderr.write("\nrender aborted by user\n")
        sys.exit(1)
    sys.stderr.write("\n")

    if a.hdr:
        imaging.save_npy(img, a.hdr)
    try:
        out = imaging.save_png((a.grade(img) * 255).astype(np.uint8), a.out)
    except ImportError:
        sys.exit("pillow not installed; rerun with --hdr to save raw .npy")
    print(f"wrote {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
