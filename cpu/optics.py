"""Dispersion and the Fresnel/refraction pair."""
import numpy as np

from cpu.spectrum import F, LMIN, LMAX

# --------------------------------------------------------------- dispersion
ND, ABBE = 2.417, 55.0


def cauchy_coeffs(fire=1.0):
    """Cauchy A, B from catalogue nd / Abbe number. `fire` exaggerates B only,
    leaving the reference index at 587.6 nm unchanged."""
    b = (ND - 1.0) / (ABBE * 1.9107) * fire
    a = ND - b * 2.8963
    return F(a), F(b)


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
