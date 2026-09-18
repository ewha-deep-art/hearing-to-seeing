"""Converting between OKLCH and the BGR hex ASS wants.

OKLCH is used rather than HSL because its lightness is perceptual: at a fixed
L every hue is about equally bright, so the subtitle stays as readable in
yellow as in blue. HSL's L does not hold that — its yellow and blue at the
same value differ enormously in how bright they look, which is exactly the
property the design depends on.

The colour model is therefore: **fix L, let H run free.** Legibility stops
being a per-colour question, and hue is left as a whole circle of choices
rather than a short list.

Chroma cannot be fixed the same way, because the sRGB gamut is not a cylinder
— how saturated a colour can be at a given lightness depends on its hue
(yellow reaches much further than blue). `max_chroma()` finds that boundary
per hue so colours stay as vivid as they can be without leaving the gamut.
"""

import math
from functools import lru_cache

# --- OKLab matrices (Björn Ottosson) -----------------------------------------

def _linear_to_srgb(x: float) -> float:
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * (x ** (1 / 2.4)) - 0.055


def _srgb_to_linear(x: float) -> float:
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def oklab_to_linear_srgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3

    return (
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def linear_srgb_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_, m_, s_ = _cbrt(l), _cbrt(m), _cbrt(s)

    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


# --- gamut -------------------------------------------------------------------

_GAMUT_EPSILON = 1e-6
_MAX_SEARCH_CHROMA = 0.5   # beyond any chroma sRGB can hold
_SEARCH_STEPS = 24         # binary search depth; ~3e-8 precision


def in_gamut(L: float, C: float, hue: float) -> bool:
    r, g, b = _to_linear(L, C, hue)
    return all(-_GAMUT_EPSILON <= v <= 1 + _GAMUT_EPSILON for v in (r, g, b))


@lru_cache(maxsize=4096)
def max_chroma(L: float, hue: float) -> float:
    """The most saturated this hue can be at this lightness and stay in sRGB.

    Binary search rather than a closed form: the gamut boundary in OKLCH has
    no simple analytic shape, and at a few dozen calls per run the cost does
    not matter.
    """
    low, high = 0.0, _MAX_SEARCH_CHROMA
    for _ in range(_SEARCH_STEPS):
        mid = (low + high) / 2
        if in_gamut(L, mid, hue):
            low = mid
        else:
            high = mid
    return low


def _to_linear(L: float, C: float, hue: float) -> tuple[float, float, float]:
    radians = math.radians(hue)
    return oklab_to_linear_srgb(L, C * math.cos(radians), C * math.sin(radians))


# --- conversion --------------------------------------------------------------

def oklch_to_rgb(L: float, C: float, hue: float) -> tuple[int, int, int]:
    """OKLCH to 8-bit sRGB, clamping anything just outside the gamut."""
    channels = _to_linear(L, C, hue)
    return tuple(
        round(255 * _linear_to_srgb(min(1.0, max(0.0, v)))) for v in channels
    )


def rgb_to_oklch(r: int, g: int, b: int) -> tuple[float, float, float]:
    """8-bit sRGB to OKLCH. Hue is in degrees, 0–360.

    Needed to read a colour *out* of an image — the clothing colour sampled
    from a frame arrives as RGB and has to become a hue before it can be used
    as a preference.
    """
    L, a, b_ = linear_srgb_to_oklab(
        *(_srgb_to_linear(v / 255) for v in (r, g, b))
    )
    chroma = math.hypot(a, b_)
    hue = math.degrees(math.atan2(b_, a)) % 360
    return L, chroma, hue


def to_ass(r: int, g: int, b: int) -> str:
    """8-bit RGB to an ASS colour literal.

    ASS stores colours byte-reversed relative to RGB — &HAABBGGRR — so
    #E69F00 is written &H00009FE6.
    """
    return f"&H00{b:02X}{g:02X}{r:02X}"


def from_ass(value: str) -> tuple[int, int, int]:
    """The inverse of `to_ass`."""
    digits = value.removeprefix("&H").removesuffix("&")
    digits = digits[-6:].rjust(6, "0")
    b, g, r = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
    return r, g, b


def hue_to_ass(hue: float, lightness: float, chroma_ratio: float = 1.0) -> str:
    """The colour for a hue, at `chroma_ratio` of the most this hue can hold.

    Chroma is given as a fraction of the per-hue maximum rather than an
    absolute value, so one ratio means the same "how vivid" across the whole
    circle even though the gamut allows far more of some hues than others.
    """
    hue %= 360
    chroma = max_chroma(lightness, hue) * chroma_ratio
    return to_ass(*oklch_to_rgb(lightness, chroma, hue))
