"""The 57-facet round brilliant, its spin, and ray-mesh intersection.

TRI/V0/E1/E2/NRM are rebuilt by set_spin(), so importers must read them through
the module (geometry.NRM), never bind them with `from ... import NRM`.
"""
import numpy as np

from cpu.spectrum import F

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


BS_C = np.array([0.0, -0.08, 0.0], F)
BS_R = F(0.60)
TRI_BASE = build_gem()


def _gem_arrays(tri):
    v0 = tri[:, 0]
    e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[np.einsum('ij,ij->i', n, tri.mean(1) - BS_C) < 0] *= -1.0
    return v0, e1, e2, n


def set_spin(deg):
    """Rotate the gem about its own axis, leaving camera, lights and floor put.

    This is not the same as moving the camera by the same angle, and the
    difference is the whole point of a turntable shot. Orbiting the camera
    keeps the stone's orientation to the lights fixed, so every facet keeps
    returning the same light and the fire pattern barely moves -- you are only
    changing which side you look at. Spinning the stone sweeps each facet
    through the light directions, which is what makes the flashes fire, die and
    change colour.

    The rotation axis is +Y, which the bounding sphere is centred on, so
    BS_C and BS_R need no adjustment.
    """
    global TRI, V0, E1, E2, NRM, SPIN
    SPIN = float(deg)
    if SPIN % 360.0 == 0.0:
        TRI = TRI_BASE                       # exact identity, no rounding
    else:
        t = np.radians(SPIN)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], F)
        TRI = (TRI_BASE @ R.T).astype(F)
    V0, E1, E2, NRM = _gem_arrays(TRI)


SPIN = 0.0
set_spin(0.0)


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
