"""Working colour spaces: which primaries the render accumulates in, and
the trip back to sRGB at the end of the grade."""
import numpy as np

from cpu import spectrum

F = np.float32

LUMA = np.array([0.2126, 0.7152, 0.0722])

XYZ_TO_REC2020 = np.array([[1.7166511880, -0.3556707838, -0.2533662814],
                           [-0.6666843518, 1.6164812366, 0.0157685458],
                           [0.0176398574, -0.0427706133, 0.9421031212]])

# Authored in sRGB, converted into the working space at pack time so the studio
# keeps its intended colour whatever primaries the render runs on.
SKY_TINT = (0.78, 0.86, 1.00)
SKY_GROUND = (0.10, 0.075, 0.055)


def _white_norm(m):
    """Integral of the CIE weights over the sampled band, in the space `m`.
    Dividing by it makes a flat spectrum render as exactly (1,1,1). Computed in
    float32 exactly as spectrum._white_norm does, so the sRGB case reproduces
    spectrum.WNORM bit for bit and the rollback identity survives."""
    lam = np.linspace(spectrum.LMIN, spectrum.LMAX, 4096, dtype=F)
    return (spectrum.cie_xyz(lam) @ np.asarray(m, F).T).mean(0).astype(F)


class WorkingSpace:
    """Primaries the renderer accumulates in, plus the trip back to sRGB."""

    def __init__(self, name, xyz_to_rgb, luma=None):
        self.name = name
        self.xyz_to_rgb = np.asarray(xyz_to_rgb, np.float64)
        self.wnorm = _white_norm(xyz_to_rgb)
        rgb_to_xyz = np.linalg.inv(self.xyz_to_rgb)
        # Luminance weights of this space are the Y row of its RGB->XYZ matrix.
        luma = np.asarray(luma if luma is not None else rgb_to_xyz[1])
        self.luma = luma / luma.sum()
        srgb = np.asarray(spectrum.XYZ_TO_RGB, np.float64)
        # None means "already sRGB": skipping the matmul rather than multiplying
        # by a nearly-identity matrix is what keeps the rollback bit-exact.
        self.to_srgb = None if name == "srgb" else srgb @ rgb_to_xyz
        if self.to_srgb is not None:
            # Both spaces are D65, so (1,1,1) must map to (1,1,1) -- the rows
            # have to sum to 1. Composed from published matrices they sum to
            # 1 +/- 3e-4, because the sRGB matrix in use is the 5-digit rounded
            # one and does not hit D65 exactly on its own. Left alone that error
            # tints every neutral in the frame and fails verification #1, so
            # normalise the rows: it fixes the white point and moves the
            # primaries by less than the rounding already present.
            self.to_srgb = self.to_srgb / self.to_srgb.sum(-1, keepdims=True)
        self.from_srgb_m = (None if self.to_srgb is None
                            else np.linalg.inv(self.to_srgb))

    def from_srgb(self, c):
        c = np.asarray(c, np.float64)
        return c if self.from_srgb_m is None else c @ self.from_srgb_m.T

    def __str__(self):
        return self.name


# LUMA is passed explicitly for sRGB: inverting the rounded matrix gives
# (0.21264, 0.71517, 0.07219), which is not the constant the spec and the old
# build use, and the difference is enough to break the bit-exact rollback.
SPACES = {
    "srgb": lambda: WorkingSpace("srgb", spectrum.XYZ_TO_RGB, LUMA),
    "rec2020": lambda: WorkingSpace("rec2020", XYZ_TO_REC2020),
}


def working_space(name):
    return SPACES[name]()
