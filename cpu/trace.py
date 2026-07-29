"""The path tracer: one randomly sampled wavelength per ray, and the
process pool that fans samples across cores."""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from cpu import geometry, lighting
from cpu.optics import cauchy_coeffs, fresnel, refract
from cpu.spectrum import F, LMIN, LMAX, rgb_to_spectral, spectral_weight

# ------------------------------------------------------------------- render
def render(W, H, spp, bounces, seed, fire, azim, elev, dist, fov, chunk,
           quiet=False, progress=None, normalize=True):
    A_C, B_C = cauchy_coeffs(fire)
    rng = np.random.default_rng(seed)
    az, el = np.radians(azim), np.radians(elev)
    eye = np.array([dist * np.cos(el) * np.sin(az), dist * np.sin(el),
                    dist * np.cos(el) * np.cos(az)], F)
    look = np.array([0.0, -0.035, 0.0], F)
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    # world-up as the roll reference, but that degenerates when looking
    # straight down (top view): fall back to +Z so `right` stays well-defined.
    world_up = np.array([0, 1, 0], F)
    if abs(float(fwd @ world_up)) > 0.999:
        world_up = np.array([0, 0, 1], F)
    right = np.cross(fwd, world_up)
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
                rad = rgb_to_spectral(lighting.env(dirs, prim), lam[where])
                np.add.at(img, ids[where], wg[where] * rad[:, None])

            t, ti = geometry.hit_gem(o, dd, F(1e-4))
            deposit(~np.isfinite(t), dd[~np.isfinite(t)], True)
            keep = np.isfinite(t)
            o, dd, ng, lam, wg, ids = (o[keep], dd[keep], ng[keep], lam[keep],
                                       wg[keep], ids[keep])
            t, ti = t[keep], ti[keep]
            if len(o) == 0:
                continue

            p = o + t[:, None] * dd
            nn = geometry.NRM[ti]
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
                t, ti = geometry.hit_gem(o, dd, F(2e-4))
                live = np.isfinite(t)
                o, dd, ng, lam, wg, ids = (o[live], dd[live], ng[live],
                                           lam[live], wg[live], ids[live])
                t, ti = t[live], ti[live]
                if len(o) == 0:
                    break
                p = o + t[:, None] * dd
                nn = geometry.NRM[ti]
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
    img = img.reshape(H, W, 3)
    return (img / spp) if normalize else img


def render_sum_worker(args):
    (W, H, spp, bounces, seed, fire, azim, elev, dist, fov, chunk,
     rig, ambient, nlights, spin) = args
    lighting.apply_scene(rig, ambient, nlights, spin)    # a fresh process has defaults
    return render(W, H, spp, bounces, seed, fire, azim, elev, dist, fov, chunk,
                  quiet=True, progress=None, normalize=False)


def render_parallel(W, H, spp, bounces, seed, fire, azim, elev, dist, fov,
                    chunk, jobs, progress=None, rig=None, ambient=None,
                    nlights=None, spin=None):
    scene = (lighting.RIG if rig is None else rig,
             lighting.AMBIENT if ambient is None else ambient,
             lighting.NLIGHTS if nlights is None else nlights,
             geometry.SPIN if spin is None else spin)
    if jobs <= 1 or spp <= 1:
        lighting.apply_scene(*scene)
        return render(W, H, spp, bounces, seed, fire, azim, elev, dist, fov,
                      chunk, quiet=False, progress=progress, normalize=True)
    jobs = min(jobs, os.cpu_count() or 1)
    n_tasks = min(spp, jobs * 8)
    counts = [(spp // n_tasks) + (1 if i < spp % n_tasks else 0)
              for i in range(n_tasks)]
    tasks = [(W, H, c, bounces, seed + i, fire, azim, elev, dist, fov, chunk)
             + scene
             for i, c in enumerate(counts) if c > 0]
    if len(tasks) == 1:
        result = render_sum_worker(tasks[0])
        if progress is not None:
            progress(spp, spp)
        return result / spp
    t0 = time.time()
    done = 0
    parts = []
    executor = ProcessPoolExecutor(max_workers=jobs)
    try:
        futures = {executor.submit(render_sum_worker, task): task for task in tasks}
        for future in as_completed(futures):
            part = future.result()
            parts.append(part)
            done += futures[future][2]
            # progress used to be treated as a flag and never called, while this
            # loop formatted a fixed line of its own -- so a caller's line, the
            # animation's whole-sequence ETA among them, could never appear.
            # Both call sites pass a callback and both format the same way the
            # old line did, so nothing changes for a single render.
            if progress is not None:
                progress(done, spp)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        if progress is not None:
            sys.stderr.write("\n")
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    if progress is not None:
        sys.stderr.write("\n")
    return np.sum(parts, axis=0) / spp
