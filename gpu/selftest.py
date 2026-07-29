"""The spec's verification suite, and CPU/GPU agreement."""
import time

import numpy as np

from cpu import imaging, lighting, spectrum, trace

from gpu import colour, grading
from gpu.renderer import F

def selftest(gpu=None):
    """The spec's verification section, plus the precondition check."""
    rng = np.random.default_rng(0)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("1. neutral preservation")
    n = np.repeat(np.geomspace(1e-3, 400.0, 64)[:, None], 3, 1)[None]
    for name in colour.SPACES:
        sp = colour.working_space(name)
        for tm, cb, kc in ((1.0, 1.35, 0.85), (0.0, 1.0, 0.0), (1.0, 2.0, 1.0)):
            out = grading.Grade(1.0, tm, cb, kc, space=sp)(n, encode=False)
            err = float(np.abs(out - out.mean(-1, keepdims=True)).max())
            check(f"neutral stays neutral in {name} "
                  f"(tm={tm} boost={cb} keep={kc})", err < 1e-6,
                  f"max channel spread {err:.2e}")

    print("2. oklab round trip")
    for lo, hi in ((0.0, 1.0), (0.0, 50.0)):
        c = rng.uniform(lo, hi, (4096, 3))
        err = float(np.abs(grading.oklab_to_linear(grading.linear_to_oklab(c)) - c).max())
        check(f"round trip on [{lo},{hi}]", err < 1e-6, f"max abs err {err:.2e}")
    for name in colour.SPACES:                      # and with the primaries folded in
        fwd, inv = grading._oklab_matrices(colour.working_space(name))
        c = rng.uniform(0.0, 50.0, (4096, 3))
        err = float(np.abs(grading.oklab_to_linear(grading.linear_to_oklab(c, fwd), inv) - c).max())
        check(f"round trip through {name} basis", err < 1e-6,
              f"max abs err {err:.2e}")

    print("3. reference pixel  linear (100, 5, 2) @ exposure 0.6")
    px = np.array([[100.0, 5.0, 2.0]])
    vivid = grading.Grade(0.6, 1.0, 1.0, 1.0)(px, encode=False)[0]
    white = grading.Grade(0.6, 1.0, 1.0, 0.0)(px, encode=False)[0]
    check("keepChroma=1 -> (1.000, 0.050, 0.020)",
          np.allclose(vivid, [1.0, 0.05, 0.02], atol=1e-3),
          f"got ({vivid[0]:.3f}, {vivid[1]:.3f}, {vivid[2]:.3f})")
    check("keepChroma=0 -> (1.000, 1.000, 1.000)",
          np.allclose(white, [1.0, 1.0, 1.0], atol=1e-3),
          f"got ({white[0]:.3f}, {white[1]:.3f}, {white[2]:.3f})")

    print("4. gamut closure, fuzzed over [-100, 500]")
    f = rng.uniform(-100.0, 500.0, (200000, 3))
    for kc in (0.0, 0.5, 1.0):
        o = grading.fit_gamut(f, kc)
        check(f"output in [0,1], finite (keep={kc})",
              bool(np.isfinite(o).all() and o.min() >= 0.0 and o.max() <= 1.0),
              f"range [{o.min():.3f}, {o.max():.3f}]")

    print("5. rollback identity")
    img = rng.uniform(-5.0, 60.0, (64, 64, 3))
    legacy = grading.Grade(0.6, 0.0, 1.0, 0.0)(img)
    old = imaging.tonemap(img, 0.6)
    err = float(np.abs(legacy - old).max())
    check("legacy settings reproduce imaging.tonemap", err == 0.0, f"max abs err {err:.2e}")
    a8 = (legacy * 255).astype(np.uint8)
    b8 = (old * 255).astype(np.uint8)
    check("identical after 8-bit encode", bool((a8 == b8).all()))

    print("6. wide-gamut working space")
    # Every monochromatic wavelength is outside both gamuts -- the locus is
    # convex-outward everywhere except at the primaries themselves -- so a
    # count of out-of-gamut wavelengths is 100% either way and says nothing.
    # What matters is how hard stage 5(a) has to pull: t is the fraction of
    # chroma it leaves behind, 1.0 meaning untouched.
    lam = np.linspace(spectrum.LMIN, spectrum.LMAX, 2048)
    xyz = spectrum.cie_xyz(lam.astype(F)).astype(np.float64)
    keep = {}
    for name in colour.SPACES:
        sp = colour.working_space(name)
        loc = xyz @ sp.xyz_to_rgb.T                  # the spectral locus itself
        L = np.maximum(loc @ sp.luma, 0.0)
        mn = loc.min(-1)
        t = np.where(mn < 0.0, np.clip(L / np.maximum(L - mn, 1e-9), 0, 1), 1.0)
        keep[name] = float(t.mean())
        print(f"        {name:8s} keeps {100 * keep[name]:4.1f}% of locus chroma "
              f"through the gamut fit")
    check("rec2020 loses less chroma on the locus than srgb",
          keep["rec2020"] > keep["srgb"],
          f"{100 * keep['rec2020']:.1f}% vs {100 * keep['srgb']:.1f}%")
    ws = colour.working_space("rec2020")
    # A neutral must survive the primaries change exactly, or the white point
    # of every studio light moves.
    err = float(np.abs(np.ones(3) @ ws.to_srgb.T - 1.0).max())
    check("rec2020 white maps to sRGB white", err < 1e-9, f"max abs err {err:.2e}")
    c = rng.uniform(-5.0, 60.0, (4096, 3))
    err = float(np.abs(ws.from_srgb(c @ ws.to_srgb.T) - c).max())
    check("primaries round trip", err < 1e-9, f"max abs err {err:.2e}")

    if gpu is not None:
        print("precondition: negatives must survive to the grading pass")
        img = gpu.render(192, 192, 200, 8, 7, 1.0, 0.0, 26.0, 2.33, 34.0, 0.25, 3)
        check("accumulator is float (not UNORM)", gpu._acc.size == gpu._acc_n * 12,
              "float32 storage buffer")
        check("negatives present in resolved HDR", img.min() < 0.0,
              f"min channel {img.min():.3f}, "
              f"{100 * (img.min(-1) < 0).mean():.2f}% of pixels")
        L = img @ grading.LUMA
        k = np.argsort(L.ravel())[-max(1, int(0.005 * L.size)):]

        def sat(x):
            x = x.reshape(-1, 3)[k]
            mx, mn = x.max(-1), x.min(-1)
            return float(np.mean(np.where(mx > 1e-9, (mx - mn) / np.maximum(mx, 1e-9), 0)))
        s_hdr = sat(img)
        s_old = sat(imaging.tonemap(img, 0.6))
        s_new = sat(grading.Grade()(img))
        print(f"        brightest 0.5% mean saturation:")
        print(f"          linear HDR            {s_hdr:.3f}")
        print(f"          per-channel ACES      {s_old:.3f}   ({100*(1-s_old/s_hdr):.0f}% lost)")
        print(f"          spec chain, srgb      {s_new:.3f}   ({100*(1-s_new/s_hdr):.0f}% lost)")
        check("spec chain retains more chroma than per-channel", s_new > s_old)

        # Same frame accumulated on wide-gamut primaries. Ranked on its own
        # luminance, since the two renders are different colour spaces and the
        # brightest pixels need not be the same set.
        wide = colour.working_space("rec2020")
        imgw = gpu.render(192, 192, 200, 8, 7, 1.0, 0.0, 26.0, 2.33, 34.0, 0.25,
                          3, space=wide)
        Lw = imgw @ wide.luma
        k = np.argsort(Lw.ravel())[-max(1, int(0.005 * Lw.size)):]
        s_wide = sat(grading.Grade(space=wide)(imgw))
        print(f"          spec chain, rec2020   {s_wide:.3f}   "
              f"({100*(1-s_wide/s_hdr):.0f}% lost)")
        print(f"        negatives reaching stage 5: "
              f"{100 * (img.min(-1) < 0).mean():.2f}% srgb, "
              f"{100 * (imgw.min(-1) < 0).mean():.2f}% rec2020")

    print(f"\n{'all checks passed' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


def compare(gpu, a):
    """Render the same frame on both backends and report agreement.

    The studio lights are tiny and very bright (~0.05% of the sphere at
    intensity 300+), so mean radiance is a high-variance estimator: a raw
    GPU-vs-CPU delta at low spp says nothing on its own. We therefore also
    render the GPU with a second seed and print that spread as the noise
    floor. A backend difference only matters if it exceeds it.
    """
    W = min(a.width, 128)
    spp = min(a.spp, 200)
    lighting.apply_scene(a.rig, a.ambient, a.lights)   # the CPU reference reads these
    if a.floor_spec:
        print("note: the CPU renderer has no floor, so --compare ignores --floor")
    print(f"comparing at {W}x{W}, {spp} spp ...")

    # The CPU reference accumulates on sRGB primaries, so pin the GPU to those
    # too: comparing a Rec.2020 render against it would measure the primaries
    # change, not backend agreement.
    srgb = colour.working_space("srgb")
    t0 = time.time()
    g = gpu.render(W, W, spp, a.bounces, a.seed, a.fire, a.azimuth, a.elevation,
                   a.distance, a.fov, a.ambient, a.lights, a.pass_spp, space=srgb)
    tg = time.time() - t0
    g2 = gpu.render(W, W, spp, a.bounces, a.seed + 1000, a.fire, a.azimuth,
                    a.elevation, a.distance, a.fov, a.ambient, a.lights,
                    a.pass_spp, space=srgb)
    t0 = time.time()
    c = trace.render(W, W, spp, a.bounces, a.seed, a.fire, a.azimuth, a.elevation,
                   a.distance, a.fov, a.chunk, quiet=True)
    tc = time.time() - t0

    gm, cm, g2m = g.mean((0, 1)), c.mean((0, 1)), g2.mean((0, 1))
    rel = lambda x, y: abs(x - y) / np.maximum(y, 1e-9) * 100
    print(f"  gpu {tg:7.2f}s   mean rgb {gm[0]:.4f} {gm[1]:.4f} {gm[2]:.4f}")
    print(f"  cpu {tc:7.2f}s   mean rgb {cm[0]:.4f} {cm[1]:.4f} {cm[2]:.4f}")
    print(f"  speedup x{tc/tg:.1f}")
    print(f"  gpu vs cpu (different backend) {rel(gm, cm).max():6.2f}%")
    print(f"  gpu vs gpu (different seed)    {rel(g2m, gm).max():6.2f}%  <- noise floor")
    print(f"  tonemapped rms difference      {np.sqrt(((imaging.tonemap(g, a.exposure) - imaging.tonemap(c, a.exposure))**2).mean()):.4f}")
    print("  the backends agree if the first number is not much above the second.")
