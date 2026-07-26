#!/usr/bin/env python3
"""
Spectral raytracer for a round-brilliant diamond.

One randomly sampled wavelength per path (380-730 nm), weighted by the CIE 1931
colour matching functions. Cauchy dispersion, exact unpolarised Fresnel,
deterministic total internal reflection.

Requires: numpy, pillow
    pip install numpy pillow

Usage:
    python spectral_diamond.py                          # 512px, 48spp
    python spectral_diamond.py -w 800 -s 200 -o gem.png # slow, clean
    python spectral_diamond.py --fire 3 --azimuth 35    # exaggerated dispersion
    python spectral_diamond.py --preview                # 192px, 12spp, ~5s
    python spectral_diamond.py --ambient 0 --lights 1   # black void, max colour
    python spectral_diamond.py --anim spin.mp4          # 60-frame rotation (ffmpeg)

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
import argparse, time, sys, os, subprocess, tempfile, shutil
import numpy as np

F = np.float32
LMIN, LMAX = 380.0, 730.0

# --------------------------------------------------------------- dispersion
ND, ABBE = 2.417, 55.0


def cauchy_coeffs(fire=1.0):
    """Cauchy A, B from catalogue nd / Abbe number. `fire` exaggerates B only,
    leaving the reference index at 587.6 nm unchanged."""
    b = (ND - 1.0) / (ABBE * 1.9107) * fire
    a = ND - b * 2.8963
    return F(a), F(b)


# ------------------------------------------------------- CIE colour matching
def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    t = (x - mu) / s
    return np.exp(-0.5 * t * t)


def cie_xyz(lam):
    """Wyman, Sloan & Shirley (2013) multi-lobe Gaussian fits to CIE 1931."""
    x = (1.056 * _g(lam, 599.8, 37.9, 31.0)
         + 0.362 * _g(lam, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam, 501.1, 20.4, 26.2))
    y = (0.821 * _g(lam, 568.8, 46.9, 40.5)
         + 0.286 * _g(lam, 530.9, 16.3, 31.1))
    z = (1.217 * _g(lam, 437.0, 11.8, 36.0)
         + 0.681 * _g(lam, 459.0, 26.0, 13.8))
    return np.stack([x, y, z], -1)


XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]], F)


def _white_norm():
    """Integral of the CIE weights over the sampled band. Dividing by this makes
    a flat spectrum render as exactly (1,1,1) -- without it everything goes
    green, because y-bar dominates."""
    lam = np.linspace(LMIN, LMAX, 4096, dtype=F)
    rgb = cie_xyz(lam) @ XYZ_TO_RGB.T
    return rgb.mean(0).astype(F)


WNORM = _white_norm()


def spectral_weight(lam):
    """RGB weight for a uniformly sampled wavelength, normalised to white."""
    return ((cie_xyz(lam) @ XYZ_TO_RGB.T) / WNORM).astype(F)


def rgb_to_spectral(rgb, lam):
    """Sample an RGB radiance at one wavelength. Three smooth windows forming a
    partition of unity, so a neutral RGB gives a flat spectrum exactly."""
    t = np.clip((lam - LMIN) / (LMAX - LMIN), 0.0, 1.0)
    b = 0.5 * (1.0 + np.cos(np.pi * np.clip((t - 0.15) / 0.25, 0, 1)))
    r = 0.5 * (1.0 - np.cos(np.pi * np.clip((t - 0.55) / 0.25, 0, 1)))
    g = 1.0 - b - r
    return (rgb[:, 0] * r + rgb[:, 1] * g + rgb[:, 2] * b).astype(F)


# ----------------------------------------------------------------- geometry
def build_gem():
    """57-facet round brilliant from cut proportions (girdle diameter 1.0)."""
    rt, yt = 0.265, 0.162
    rp, yp = 0.370, 0.100
    rg, ygt, ygb = 0.500, 0.015, -0.015
    rn, yn = 0.280, -0.200
    yc = -0.430

    a16 = np.arange(16) * (2 * np.pi / 16)
    a8 = np.arange(8) * (2 * np.pi / 8)
    ring = lambda r, y, a: np.stack([r * np.cos(a), np.full(len(a), y),
                                     r * np.sin(a)], 1)
    T = ring(rt, yt, a8)
    P = ring(rp, yp, a8 + np.pi / 8)
    Gt = ring(rg, ygt, a16)
    Gb = ring(rg, ygb, a16)
    N = ring(rn, yn, a8 + np.pi / 8)
    C = np.array([0.0, yc, 0.0])

    tris = []
    quad = lambda a, b, c, d: (tris.append([a, b, c]), tris.append([a, c, d]))

    for j in range(1, 7):
        tris.append([T[0], T[j], T[j + 1]])                     # table
    for j in range(8):
        k = (j + 1) % 8
        tris.append([T[j], T[k], P[j]])                         # star
        quad(T[j], P[(j - 1) % 8], Gt[2 * j], P[j])             # bezel kite
        tris.append([P[j], Gt[2 * j], Gt[2 * j + 1]])           # upper girdle
        tris.append([P[j], Gt[2 * j + 1], Gt[(2 * j + 2) % 16]])
        tris.append([Gb[2 * j], Gb[2 * j + 1], N[j]])           # lower girdle
        tris.append([Gb[2 * j + 1], Gb[(2 * j + 2) % 16], N[j]])
        quad(Gb[2 * j], N[j], C, N[(j - 1) % 8])                # pavilion main
    for k in range(16):
        m = (k + 1) % 16
        quad(Gt[k], Gt[m], Gb[m], Gb[k])                        # girdle band
    return np.array(tris, dtype=F)


TRI = build_gem()
V0 = TRI[:, 0]
E1, E2 = TRI[:, 1] - V0, TRI[:, 2] - V0
NRM = np.cross(E1, E2)
NRM /= np.linalg.norm(NRM, axis=1, keepdims=True)
BS_C = np.array([0.0, -0.08, 0.0], F)
NRM[np.einsum('ij,ij->i', NRM, TRI.mean(1) - BS_C) < 0] *= -1.0
BS_R = F(0.60)


def hit_gem(org, dir, tmin):
    """Nearest triangle hit, with a bounding-sphere prefilter."""
    n = len(org)
    t = np.full(n, np.inf, F)
    idx = np.zeros(n, np.int32)
    oc = org - BS_C
    b = np.einsum('ij,ij->i', oc, dir)
    c = np.einsum('ij,ij->i', oc, oc) - BS_R * BS_R
    disc = b * b - c
    cand = np.nonzero((disc > 0) &
                      ((-b + np.sqrt(np.maximum(disc, 0))) > tmin))[0]
    if len(cand) == 0:
        return t, idx
    o, d = org[cand], dir[cand]
    pv = np.cross(d[:, None, :], E2[None])
    det = np.einsum('mtj,tj->mt', pv, E1)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o[:, None, :] - V0[None]
    u = np.einsum('mtj,mtj->mt', tv, pv) * inv
    qv = np.cross(tv, E1[None])
    v = np.einsum('mj,mtj->mt', d, qv) * inv
    tt = np.einsum('mtj,tj->mt', qv, E2) * inv
    good = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (tt > tmin)
    tt = np.where(good, tt, np.inf)
    k = np.argmin(tt, axis=1)
    t[cand] = tt[np.arange(len(cand)), k]
    idx[cand] = k
    return t, idx


# -------------------------------------------------------------- environment
# (direction, angular width, intensity). Small and bright: a broad soft light
# widens the dispersion fan until the colours overlap back into white.
LIGHTS = [((0.45, 0.82, 0.35), 0.0011, 300.0),
          ((-0.62, 0.70, 0.35), 0.0015, 210.0),
          ((0.10, 0.93, -0.35), 0.0008, 430.0),
          ((-0.25, 0.35, 0.90), 0.0038, 70.0),
          ((0.80, 0.25, -0.55), 0.0026, 100.0)]
LIGHTS = [(np.array(d, F) / np.linalg.norm(d), F(w), F(i)) for d, w, i in LIGHTS]


AMBIENT = 0.25   # scales the sky/ground fill only; lights are unaffected
NLIGHTS = 3


def env(d, primary=False):
    y = d[:, 1:2]
    sky = np.where(y > 0, 0.28 + 0.42 * np.clip(y, 0, 1),
                   0.16 * np.exp(4.0 * np.clip(y, -1, 0)))
    col = sky * np.array([[0.78, 0.86, 1.00]], F)
    col = (col + (y < -0.05) * np.array([[0.10, 0.075, 0.055]], F)) * AMBIENT
    if primary:
        return (col * 0.16).astype(F)
    for ld, w, inten in LIGHTS[:NLIGHTS]:
        s = np.einsum('ij,j->i', d, ld)[:, None]
        col = col + inten * np.exp(-(1.0 - np.clip(s, -1, 1)) / w)
    return col.astype(F)


# ------------------------------------------------------------------ shading
def fresnel(cosi, eta):
    sin2 = eta * eta * (1.0 - cosi * cosi)
    tir = sin2 >= 1.0
    cost = np.sqrt(np.clip(1.0 - sin2, 0.0, 1.0))
    rs = (eta * cosi - cost) / np.maximum(eta * cosi + cost, 1e-7)
    rp = (cosi - eta * cost) / np.maximum(cosi + eta * cost, 1e-7)
    return np.where(tir, 1.0, np.clip(0.5 * (rs * rs + rp * rp), 0, 1)), tir


def refract(I, N, eta, cosi):
    k = np.maximum(1.0 - eta * eta * (1.0 - cosi * cosi), 0.0)
    return eta[:, None] * I + (eta * cosi - np.sqrt(k))[:, None] * N


# ------------------------------------------------------------------- render
def render(W, H, spp, bounces, seed, fire, azim, elev, dist, fov, chunk,
           quiet=False, progress=None):
    A_C, B_C = cauchy_coeffs(fire)
    rng = np.random.default_rng(seed)
    az, el = np.radians(azim), np.radians(elev)
    eye = np.array([dist * np.cos(el) * np.sin(az), dist * np.sin(el),
                    dist * np.cos(el) * np.cos(az)], F)
    look = np.array([0.0, -0.035, 0.0], F)
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 1, 0], F))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    scale = F(np.tan(np.radians(fov * 0.5)))

    img = np.zeros((H * W, 3), np.float64)
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    px = px.ravel().astype(F)
    py = py.ravel().astype(F)
    npix = W * H
    t0 = time.time()

    for s in range(spp):
        sx = (2.0 * (px + rng.random(npix).astype(F)) / W - 1.0) * scale
        sy = (1.0 - 2.0 * (py + rng.random(npix).astype(F)) / H) * scale
        d0 = fwd[None] + sx[:, None] * right[None] + sy[:, None] * up[None]
        d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
        lam0 = (LMIN + (LMAX - LMIN) * rng.random(npix)).astype(F)
        um = lam0 * F(1e-3)
        ng0 = A_C + B_C / (um * um)
        wg0 = spectral_weight(lam0)

        for c0 in range(0, npix, chunk):
            sl = slice(c0, min(c0 + chunk, npix))
            m = sl.stop - sl.start
            o = np.repeat(eye[None], m, 0)
            dd, ng, lam, wg = d0[sl].copy(), ng0[sl], lam0[sl], wg0[sl]
            ids = np.arange(sl.start, sl.stop)

            def deposit(where, dirs, prim=False):
                if not where.any():
                    return
                rad = rgb_to_spectral(env(dirs, prim), lam[where])
                np.add.at(img, ids[where], wg[where] * rad[:, None])

            t, ti = hit_gem(o, dd, F(1e-4))
            deposit(~np.isfinite(t), dd[~np.isfinite(t)], True)
            keep = np.isfinite(t)
            o, dd, ng, lam, wg, ids = (o[keep], dd[keep], ng[keep], lam[keep],
                                       wg[keep], ids[keep])
            t, ti = t[keep], ti[keep]
            if len(o) == 0:
                continue

            p = o + t[:, None] * dd
            nn = NRM[ti]
            ci = -np.einsum('ij,ij->i', nn, dd)
            nn = np.where((ci < 0)[:, None], -nn, nn)
            ci = np.abs(ci)
            eta = 1.0 / ng
            R, _ = fresnel(ci, eta)
            ext = rng.random(len(o)).astype(F) < R
            deposit(ext, dd[ext] + 2.0 * ci[ext][:, None] * nn[ext])

            ib = ~ext
            dd = refract(dd[ib], nn[ib], eta[ib], ci[ib])
            dd /= np.linalg.norm(dd, axis=1, keepdims=True)
            o, ng, lam, wg, ids = p[ib], ng[ib], lam[ib], wg[ib], ids[ib]

            for _ in range(bounces + 1):
                if len(o) == 0:
                    break
                t, ti = hit_gem(o, dd, F(2e-4))
                live = np.isfinite(t)
                o, dd, ng, lam, wg, ids = (o[live], dd[live], ng[live],
                                           lam[live], wg[live], ids[live])
                t, ti = t[live], ti[live]
                if len(o) == 0:
                    break
                p = o + t[:, None] * dd
                nn = NRM[ti]
                sgn = np.einsum('ij,ij->i', nn, dd)
                nn = np.where((sgn < 0)[:, None], -nn, nn)
                ci = np.abs(np.einsum('ij,ij->i', nn, dd))
                R, tir = fresnel(ci, ng)
                out = (rng.random(len(o)).astype(F) >= R) & (~tir)
                if out.any():
                    od = refract(dd[out], -nn[out], ng[out], ci[out])
                    od /= np.linalg.norm(od, axis=1, keepdims=True)
                    deposit(out, od)
                ins = ~out
                dd = dd[ins] - 2.0 * np.einsum(
                    'ij,ij->i', nn[ins], dd[ins])[:, None] * nn[ins]
                o, ng, lam, wg, ids = (p[ins], ng[ins], lam[ins], wg[ins],
                                       ids[ins])
        if progress is not None:
            progress(s + 1, spp)
        elif not quiet:
            el_s = time.time() - t0
            sys.stderr.write(f"\r  {s+1}/{spp} spp  {el_s:6.1f}s "
                             f"(eta {el_s/(s+1)*(spp-s-1):6.1f}s)")
            sys.stderr.flush()
    if progress is None and not quiet:
        sys.stderr.write("\n")
    return (img / spp).reshape(H, W, 3)


def tonemap(x, exposure):
    x = np.maximum(x, 0.0) * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1) ** (1 / 2.2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-w", "--width", type=int, default=512)
    p.add_argument("-s", "--spp", type=int, default=48,
                   help="samples per pixel (one wavelength each)")
    p.add_argument("-b", "--bounces", type=int, default=8,
                   help="max internal reflections")
    p.add_argument("-o", "--out", default="diamond.png")
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
    p.add_argument("--lights", type=int, default=3, choices=range(1, 6),
                   help="how many of the 5 studio lights to use")
    p.add_argument("--chunk", type=int, default=24000,
                   help="rays per batch; lower it if you run out of memory")
    p.add_argument("--hdr", help="also save raw linear radiance as .npy")
    p.add_argument("--preview", action="store_true",
                   help="fast 192px / 12spp preview")
    p.add_argument("--anim", metavar="OUT.mp4",
                   help="render a rotation animation to this .mp4 (needs ffmpeg)")
    p.add_argument("--anim-frames", type=int, default=60,
                   help="number of frames for --anim (default 60)")
    p.add_argument("--anim-step", type=float, default=0.1,
                   help="azimuth increment per frame in degrees (default 0.1)")
    p.add_argument("--fps", type=int, default=30,
                   help="frame rate of the --anim .mp4 (default 30)")
    a = p.parse_args()
    if a.preview:
        a.width, a.spp = 192, 12
    global AMBIENT, NLIGHTS
    AMBIENT, NLIGHTS = a.ambient, a.lights

    if a.anim:
        animate(a)
        return

    print(f"{len(TRI)} triangles | {a.width}x{a.width} | {a.spp} spp | "
          f"{a.bounces} internal reflections | fire x{a.fire:g} | "
          f"{a.lights} lights | ambient {a.ambient:g}")
    t0 = time.time()
    img = render(a.width, a.width, a.spp, a.bounces, a.seed, a.fire,
                 a.azimuth, a.elevation, a.distance, a.fov, a.chunk)
    if a.hdr:
        np.save(a.hdr, img)
    try:
        from PIL import Image
        Image.fromarray((tonemap(img, a.exposure) * 255).astype(np.uint8)).save(a.out)
    except ImportError:
        sys.exit("pillow not installed; rerun with --hdr to save raw .npy")
    print(f"wrote {a.out} in {time.time()-t0:.1f}s")


def animate(a):
    """Render a.anim_frames frames, each rotated a.anim_step degrees in azimuth
    from the previous one, then stitch them into an .mp4 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH; install it or drop --anim")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("pillow not installed; --anim needs it to write frames")

    tmp = tempfile.mkdtemp(prefix="diamond_anim_")
    print(f"{len(TRI)} triangles | {a.width}x{a.width} | {a.spp} spp | "
          f"{a.anim_frames} frames x {a.anim_step:g} deg | fire x{a.fire:g} | "
          f"{a.lights} lights | ambient {a.ambient:g}")
    t0 = time.time()
    try:
        for i in range(a.anim_frames):
            azim = a.azimuth + i * a.anim_step

            def show(done, total, i=i):
                sys.stderr.write(f"\r  frame {i+1}/{a.anim_frames} completed "
                                 f"{100 * done // total:3d}%  ({time.time()-t0:6.1f}s)")
                sys.stderr.flush()

            img = render(a.width, a.width, a.spp, a.bounces, a.seed, a.fire,
                         azim, a.elevation, a.distance, a.fov, a.chunk,
                         progress=show)
            frame = tonemap(img, a.exposure)
            Image.fromarray((frame * 255).astype(np.uint8)).save(
                os.path.join(tmp, f"frame_{i:05d}.png"))
            sys.stderr.write("\n")
        subprocess.run(
            [ffmpeg, "-y", "-framerate", str(a.fps),
             "-i", os.path.join(tmp, "frame_%05d.png"),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", "-movflags", "+faststart", a.anim],
            check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.anim} ({a.anim_frames} frames) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()