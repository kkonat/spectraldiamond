"""Lighting rigs, the ambient fill, and the environment lookup.

LIGHTS/RIG/AMBIENT/NLIGHTS are rebound by apply_scene(), so importers must read
them through the module (lighting.NLIGHTS), not bind them by name.
"""
import numpy as np

from cpu import geometry
from cpu.spectrum import F

# -------------------------------------------------------------- environment
# Lighting rigs: (direction, angular width w, intensity), plus how many of the
# five to use by default.
#
# w is not an angle. The falloff is exp(-(1-cos t)/w) and 1-cos t ~ t^2/2, so
# the source is a Gaussian of sigma = sqrt(w) radians. Sizes below are quoted
# as that sigma, which is the number that matters: a source wider than the
# dispersion fan overlaps its own spectrum back into white. Measured on this
# scene, mean saturation falls monotonically with source size --
#
#     sigma   0.12   0.48   0.95   1.90   3.80   7.60 deg
#     sat     0.888  0.841  0.707  0.630  0.581  0.467
#
# -- with lit area moving the other way (0.1% to 11%). No size wins both, which
# is why the useful rigs mix hard keys for fire with broad fills for form. The
# knee is near 0.5 deg, about the angular size of the sun; going smaller buys
# little purity and costs a lot of light and a lot of samples.
RIGS = {
    # bright, brilliant, reads as a solid faceted object, almost no colour
    "softbox": ([((0.45, 0.82, 0.35), 0.0176, 18.75),
                 ((-0.62, 0.70, 0.35), 0.0240, 13.13),
                 ((0.10, 0.93, -0.35), 0.0128, 26.88),
                 ((-0.25, 0.35, 0.90), 0.0608, 4.38),
                 ((0.80, 0.25, -0.55), 0.0416, 6.25)], 3),
    # the original rig: everything at 1.6-3.5 deg, halfway between the two ends
    "studio":  ([((0.45, 0.82, 0.35), 0.0011, 300.0),
                 ((-0.62, 0.70, 0.35), 0.0015, 210.0),
                 ((0.10, 0.93, -0.35), 0.0008, 430.0),
                 ((-0.25, 0.35, 0.90), 0.0038, 70.0),
                 ((0.80, 0.25, -0.55), 0.0026, 100.0)], 3),
    # two hard keys against one broad fill: dark and contrasty, fire-leaning
    "fire":    ([((0.45, 0.82, 0.35), 0.00006875, 4800.0),
                 ((-0.62, 0.70, 0.35), 0.0135, 23.3),
                 ((0.10, 0.93, -0.35), 0.00005, 6880.0),
                 ((-0.25, 0.35, 0.90), 0.0038, 70.0),
                 ((0.80, 0.25, -0.55), 0.0026, 100.0)], 3),
    # same keys, three broad fills: bright and readable and still sparkling.
    # 3x the fiery area of "studio" for a 5% drop in per-flash saturation.
    "showcase": ([((0.45, 0.82, 0.35), 0.00006875, 4800.0),
                  ((-0.62, 0.70, 0.35), 0.0135, 23.3),
                  ((0.10, 0.93, -0.35), 0.00005, 6880.0),
                  ((-0.25, 0.35, 0.90), 0.0342, 7.8),
                  ((0.80, 0.25, -0.55), 0.0234, 11.1)], 5),
}
DEFAULT_RIG = "studio"


def _norm_rig(name):
    return [(np.array(d, F) / np.linalg.norm(d), F(w), F(i))
            for d, w, i in RIGS[name][0]]


LIGHTS = _norm_rig(DEFAULT_RIG)
RIG = DEFAULT_RIG

AMBIENT = 0.25   # scales the sky/ground fill only; lights are unaffected
NLIGHTS = RIGS[DEFAULT_RIG][1]


def apply_scene(rig=None, ambient=None, nlights=None, spin=None):
    """Set the module-level lighting state, returning the light count in use.

    These are globals because env() is called from the inner loop, and on
    Windows every worker process re-imports this module and gets the defaults
    back -- so a worker must call this itself. Passing them through the task
    tuple is what makes --rig, --ambient and --lights reach the CPU renderer's
    workers at all; before this they were set only in the parent and silently
    ignored by every parallel render."""
    global LIGHTS, RIG, AMBIENT, NLIGHTS
    if rig is not None:
        RIG, LIGHTS = rig, _norm_rig(rig)
        NLIGHTS = RIGS[rig][1]
    if ambient is not None:
        AMBIENT = ambient
    if nlights is not None:
        NLIGHTS = nlights
    if spin is not None and spin != geometry.SPIN:
        set_spin(spin)
    return NLIGHTS


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
