"""Wavelength to colour: the CIE fit, the sRGB matrix and the white norm.

The lowest layer of the package -- imports nothing else from it.
"""
import numpy as np

F = np.float32
LMIN, LMAX = 380.0, 730.0


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
