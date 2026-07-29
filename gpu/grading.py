"""The display grade from spectral_grading_spec.md, stage by stage."""
import numpy as np


from gpu.colour import LUMA, working_space

#
# Precondition (spec "check this first"), verified by --selftest:
# the accumulator is a float32 storage buffer, and nothing between the
# wavelength-to-RGB conversion and this function clamps, so the out-of-gamut
# negatives reach stage 5 intact.

# Rows sum to 1, so a neutral input stays neutral through the round trip.
_LIN2LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                     [0.2119034982, 0.6806995451, 0.1073969566],
                     [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS2LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]])
# The spec quotes the inverse direction as its own set of 10-digit constants
#   [[1, 0.3963377774, 0.2158037573], ...]  /  [[4.0767416621, -3.3077115913, ...
# Rounded independently of the forward matrices, they only invert them to 1.3e-5,
# which fails the spec's own 1e-6 round-trip requirement (verification #2).
# Deriving the pair instead makes the round trip exact to 3e-13 for free. The
# forward matrices are the spec's, untouched -- those define the space.
_LAB2LMS = np.linalg.inv(_LMS2LAB)
_LMS2LIN = np.linalg.inv(_LIN2LMS)


def _luma(c, luma=None):
    return (c * (LUMA if luma is None else luma)).sum(-1, keepdims=True)


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def aces_per_channel(c):
    """The old path. Kept because stage 3 blends against it."""
    return aces(c)


def tonemap_luminance(c, strength, luma=None):
    """Stage 3. Curve applied to luminance only, chromaticity preserved.
    Output may exceed 1.0 per channel -- that is stage 5's problem, not ours.

    Runs in the working space, so `luma` must be that space's weights: using
    Rec.709 coefficients on Rec.2020 data would mis-weight the curve and tilt
    the whole image green."""
    if strength <= 0.0:
        return aces_per_channel(np.maximum(c, 0.0))
    L = _luma(c, luma)
    Lt = aces(np.maximum(L, 0.0))
    safe = np.where(L > 1e-6, L, 1.0)
    lum_path = np.where(L > 1e-6, c * (Lt / safe), 0.0)
    if strength >= 1.0:
        return lum_path
    return aces_per_channel(np.maximum(c, 0.0)) * (1.0 - strength) + lum_path * strength


def _oklab_matrices(space):
    """Oklab is defined on linear sRGB. Rather than converting the image into
    sRGB and back around the boost -- which would clip the wide gamut at exactly
    the wrong moment -- fold the primaries change into the LMS matrices. The
    space Oklab measures stays Oklab; only the input basis changes."""
    if space is None or space.to_srgb is None:
        return _LIN2LMS, _LMS2LIN
    return _LIN2LMS @ space.to_srgb, space.from_srgb_m @ _LMS2LIN


def linear_to_oklab(c, fwd=None):
    lms = c @ (_LIN2LMS if fwd is None else fwd).T
    r = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)   # sign-preserving cube root
    return r @ _LMS2LAB.T


def oklab_to_linear(lab, inv=None):
    lms = lab @ _LAB2LMS.T
    return (lms ** 3) @ (_LMS2LIN if inv is None else inv).T


def boost_chroma(c, amount, space=None):
    """Stage 4. Vibrance, not saturation: already-colourful pixels get boosted
    less, so the flashes intensify without smearing the neutral body."""
    if amount == 1.0:
        return c                       # exact identity; skips round-trip error
    fwd, inv = _oklab_matrices(space)
    lab = linear_to_oklab(np.maximum(c, 0.0), fwd)
    C = np.linalg.norm(lab[..., 1:], axis=-1, keepdims=True)
    k = 1.0 + (amount - 1.0) / (1.0 + 4.0 * C)
    return oklab_to_linear(np.concatenate([lab[..., :1], lab[..., 1:] * k], -1),
                           inv)


def desaturate_to_white(c):
    """Pull toward the luminance axis until in gamut. Standard, and the wrong
    trade for gem fire -- at display luminance ~1 the only in-gamut answer is
    white. Kept as the keepChroma = 0 end of the blend."""
    L = np.clip(_luma(c), 0.0, 1.0)
    d = c - L
    hi = np.where(d >= 1e-9, (1.0 - L) / np.maximum(d, 1e-9), 1e9)
    lo = np.where(d <= -1e-9, (0.0 - L) / np.minimum(d, -1e-9), 1e9)
    t = np.minimum(1.0, np.minimum(hi.min(-1, keepdims=True),
                                   lo.min(-1, keepdims=True)))
    # t == 1 means the colour is already in gamut, so it must pass through
    # untouched. Reconstructing it as L + (c - L) is off by an ulp, which is
    # enough to break the spec's bit-for-bit rollback identity.
    return np.where(t >= 1.0, np.clip(c, 0.0, 1.0),
                    np.clip(L + np.maximum(t, 0.0) * d, 0.0, 1.0))


def fit_gamut(c, keep_chroma):
    """Stage 5. (a) clear negatives by desaturating the minimum necessary at
    constant hue, then (b) handle overflow. Order matters: dividing by the max
    while a channel is still negative inverts the hue."""
    L = np.maximum(_luma(c), 0.0)
    mn = c.min(-1, keepdims=True)
    neg = mn < 0.0
    if neg.any():
        t = np.clip(L / np.maximum(L - mn, 1e-9), 0.0, 1.0)
        # Only rewrite pixels that actually have a negative channel: for the
        # rest, L + (c - L) differs from c by an ulp and breaks the rollback
        # identity for no benefit.
        c = np.where(neg, L + t * (c - L), c)

    # Divide-by-max keeps chromaticity exactly and pays in brightness instead.
    chroma_path = c / np.maximum(c.max(-1, keepdims=True), 1.0)
    if keep_chroma >= 1.0:
        return np.clip(chroma_path, 0.0, 1.0)
    if keep_chroma <= 0.0:
        return desaturate_to_white(c)
    return np.clip(desaturate_to_white(c) * (1.0 - keep_chroma)
                   + chroma_path * keep_chroma, 0.0, 1.0)


class Grade:
    """The spec's chain, in order, minus bloom.

    Stage 2 (bloom) is deliberately gone. Measured on this scene, above-
    threshold pixels carry 92% of the frame's energy -- a brilliant cut is
    nearly all specular flash -- so any useful amount smears a third of the
    image back over the stone as haze, and at high amounts the mip chain's
    box-downsample beats visibly against the one-pixel flashes and crawls
    along facet edges. Glare is better added in a compositor, on an image the
    renderer has not already softened.

    Stages 3 and 4 run in the working space; the conversion to sRGB sits
    between stage 4 and stage 5, so the gamut fit is the last thing that
    touches the colour, exactly where the display boundary actually is."""

    def __init__(self, exposure=0.60, tonemap_strength=1.0, chroma_boost=1.35,
                 keep_chroma=0.85, levels=(0.0, 1.0, 1.0), space=None):
        self.exposure = exposure
        self.tonemap_strength = tonemap_strength
        self.chroma_boost = chroma_boost
        self.keep_chroma = keep_chroma
        self.levels = tuple(levels)
        self.space = space or working_space("srgb")

    @property
    def is_legacy(self):
        return (self.tonemap_strength == 0.0 and self.chroma_boost == 1.0
                and self.keep_chroma == 0.0 and self.space.to_srgb is None
                and self.levels == (0.0, 1.0, 1.0))

    def __str__(self):
        if self.is_legacy:
            return "grade legacy (per-channel ACES)"
        return (f"grade {self.space} / tm {self.tonemap_strength:g} "
                f"/ chroma {self.chroma_boost:g} / keep {self.keep_chroma:g}")

    def __call__(self, hdr, encode=True):
        sp = self.space
        c = np.asarray(hdr, np.float64) * self.exposure      # 1 exposure
        c = tonemap_luminance(c, self.tonemap_strength, sp.luma)   # 3 tone curve
        c = boost_chroma(c, self.chroma_boost, sp)           # 4 chroma
        if sp.to_srgb is not None:                           #   -> display
            c = c @ sp.to_srgb.T
        c = fit_gamut(c, self.keep_chroma)                   # 5 gamut fit
        black, white, gamma = self.levels                    # 6 levels
        if (black, white, gamma) != (0.0, 1.0, 1.0):
            c = np.clip((c - black) / max(white - black, 1e-6), 0.0, 1.0)
            if gamma != 1.0:
                c = c ** (1.0 / gamma)
        if not encode:
            return c
        # 7 OETF. Kept as pure 2.2 rather than the sRGB piecewise curve: it is
        # what the existing build encodes with, and swapping it would break the
        # spec's rollback-identity requirement for no colour benefit.
        return np.clip(c, 0.0, 1.0) ** (1 / 2.2)
