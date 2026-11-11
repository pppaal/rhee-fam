"""Number notation shared by every user-facing surface.

One implementation of the rule that the narrative (``narrative.format_number``),
the analysis' phase reasons (``analysis._num_txt``) and the report (tiles, axis
labels, tables) all follow, so a value never reads differently on two surfaces:

* thousands separators and a precision chosen from the magnitude
  (``1,234`` / ``45.2`` / ``1.20`` / ``0.023``);
* magnitudes below :data:`SCI_SMALL` (1e-4) or from :data:`SCI_LARGE` (1e12) up
  in ``1.05×10⁻⁹`` / ``1.87×10¹³`` notation -- never Python's ``2.392e+06``, never
  a 13-digit integer;
* the typographic minus U+2212.

This module imports nothing from the package, so every module can use it.
"""

from __future__ import annotations

import math
from typing import Any, Optional

__all__ = ["MINUS", "SCI_SMALL", "SCI_LARGE", "superscript", "sci", "uses_sci", "number_text",
           "Z_SHOWN_MAX", "zscore_text"]

#: typographic minus used in every user-facing number
MINUS = "−"
#: ×10ⁿ notation below / from these magnitudes: a 13-digit integer such as 18,698,199,572,300 is
#: unreadable in prose, and a tiny value must not collapse to "0"
SCI_SMALL, SCI_LARGE = 1e-4, 1e12

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _finite(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def superscript(e: int) -> str:
    """The exponent of ``×10ⁿ`` as superscript digits: ``-9`` -> ``⁻⁹``."""
    e = int(e)
    return ("⁻" if e < 0 else "") + str(abs(e)).translate(_SUPERSCRIPT)


def sci(a: float, digits: int = 3) -> str:
    """``×10ⁿ`` notation of the magnitude ``|a|`` with ``digits`` significant digits, trailing
    zeros dropped: ``1.05×10⁻⁹``, ``4×10¹⁶``; no sign (the caller adds it).  An exponent of 0
    gives the plain mantissa."""
    mant, ex = f"{abs(float(a)):.{max(1, int(digits)) - 1}e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    e = int(ex)
    if e == 0:
        return mant
    return f"{mant}×10{superscript(e)}"


def uses_sci(a: float, scale: Any = None) -> bool:
    """Whether :func:`number_text` writes ``|a|`` in ``×10ⁿ`` notation.

    Below 1e-4 (but not 0) and from 1e12 up; also from 1,000 up when ``scale`` -- the largest
    magnitude printed next to it -- is at least 1e12, so one line keeps one notation
    (``4.61×10¹¹ → 1.51×10¹⁵``, not ``460,927,814,133 → 1.51×10¹⁵``).
    """
    a = abs(float(a))
    big = max(a, abs(_finite(scale) or 0.0))
    return 0 < a < SCI_SMALL or a >= SCI_LARGE or (a >= 1000 and big >= SCI_LARGE)


def number_text(x: Any, decimals: Optional[int] = None, signed: bool = False,
                scale: Any = None) -> Optional[str]:
    """A finite number as every surface prints it; ``None`` for None / NaN / inf.

    ``decimals=None`` picks the precision from the magnitude (``1,234`` / ``45.2`` /
    ``1.20`` / ``0.02``) and switches to ``×10ⁿ`` by :func:`uses_sci`; a fixed
    ``decimals`` keeps fixed notation.  ``signed=True`` prefixes positive values
    with ``+``; a value that rounds to 0 has no sign.
    """
    v = _finite(x)
    if v is None:
        return None
    a = abs(v)
    if decimals is None and uses_sci(a, scale):
        mant, exp = f"{a:.2e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        sign = MINUS if v < 0 else ("+" if signed else "")
        return f"{sign}{mant}×10{superscript(int(exp))}"
    strip = False
    if decimals is not None:
        d = max(0, int(decimals))
    elif a >= 1000:
        d = 0
    elif a >= 10:
        d = 1
    elif a >= 0.1:
        d = 2
    elif a == 0:
        d = 0
    else:  # small numbers: two significant digits, trailing zeros removed
        d = min(8, -int(math.floor(math.log10(a))) + 1)
        strip = True
    r = round(v, d)
    body = f"{abs(r):,.{d}f}"
    if strip and "." in body:
        body = body.rstrip("0").rstrip(".")
    if r == 0 or body.strip("0.,") == "":
        return "0" if d == 0 or strip else f"{0:.{d}f}"
    sign = MINUS if r < 0 else ("+" if signed else "")
    return sign + body


#: robust z-scores beyond this size are shown as a bound (``z > +100``): on an almost constant
#: history (a lone first order after 155 zero weeks) the residual scale is ~0 and the z-score a
#: number in the thousands ("z = +6,239.4") that says no more than "far outside anything seen"
Z_SHOWN_MAX = 100.0


def zscore_text(z: Any, label: str = "z") -> str:
    """A robust z-score as every surface prints it: ``z = +9.6``, or ``z > +100`` / ``z < −100``
    beyond :data:`Z_SHOWN_MAX`; ``label=''`` gives the bare value (``+9.6`` / ``> +100``) for a
    table cell.  A missing value prints as ``—``."""
    v = _finite(z)
    if v is None:
        return f"{label} = —" if label else "—"
    if abs(v) > Z_SHOWN_MAX:
        bound = number_text(math.copysign(Z_SHOWN_MAX, v), decimals=0, signed=True)
        op = ">" if v > 0 else "<"
        return f"{label} {op} {bound}" if label else f"{op} {bound}"
    num = number_text(v, decimals=1, signed=True)
    return f"{label} = {num}" if label else str(num)
