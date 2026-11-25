"""Display formatting for the web VM (spec D.1), mirrored function for function by ``static/js/fmt.js``.

Owner: **WP1-vm**. Also imported by WP1-server (``analyze.py``: the ``span`` of ``horizon_clamped``, the ``fetched``
time of ``naver_cached``).

Rules for every function:

* All display rounding goes through :func:`round_half_away` (exact decimal value of the double, ties away from zero).
  Never ``round()``, ``format(x, ".Nf")`` or the engine's own number helpers in ``trendai.numfmt`` (this module must
  not import that module; ``tests/test_web_fmt.py`` checks).
* Scaling by a power of ten (the compact units, the ×10ⁿ mantissa) is done on the exact decimal value as well
  (``Decimal.scaleb``), so ``99950000`` is exactly ``9995`` 만 before it is rounded -- the JS side shifts the digit
  string of the exact expansion the same way.
* The minus sign is U+2212 (:data:`MINUS`); a missing value (``None``/NaN/inf) prints :data:`DASH` (``—``).
* ``lang`` is ``"ko"`` or ``"en"``; any other value is treated as ``"ko"``.
* Every row of the D.1.7 test vectors holds for both this module and ``fmt.js``.
"""

from __future__ import annotations

import datetime as _dt
import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

__all__ = [
    "MINUS", "DASH", "MONTHS_EN",
    "round_half_away",
    "compact", "full", "value_text",
    "pct", "pct_arrow", "pct_abs", "pp",
    "prob_text", "prob_exact", "prob_labelled", "likelihood",
    "date_label", "date_short", "range_label", "datetime_label",
    "span_label", "freq_label", "freq_adjective",
]

MINUS = "−"
DASH = "—"
MONTHS_EN: tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#: value kinds printed with one fixed decimal and never compacted (D.1.3)
_INDEX_KINDS = ("relative_index", "relative_index_gt")
#: ×10ⁿ thresholds of :func:`full` (the engine's notation rule, D.1)
_SCI_SMALL, _SCI_LARGE = Decimal("1e-4"), Decimal("1e12")
#: compact units (power of ten, suffix), smallest first, and the value from which the next unit takes over
_UNITS = {"ko": ((4, "만"), (8, "억"), (12, "조")), "en": ((3, "K"), (6, "M"), (9, "B"))}
_UNIT_LIMIT = {"ko": Decimal(10000), "en": Decimal(1000)}
#: compact numbers from these magnitudes are written ×10ⁿ
_COMPACT_SCI = {"ko": Decimal("1e16"), "en": Decimal("1e12")}
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_KST = _dt.timedelta(hours=9)

_FREQ_LABELS = {"D": ("일 단위", "daily"), "W": ("주 단위", "weekly"), "M": ("월 단위", "monthly"),
                "Q": ("분기 단위", "quarterly"), "Y": ("연 단위", "yearly")}
_FREQ_ADJ_KO = {"D": "일간", "W": "주간", "M": "월간", "Q": "분기", "Y": "연간"}
_SPAN_KO = {"D": "일", "W": "주", "M": "개월", "Q": "분기", "Y": "년"}
_SPAN_EN = {"D": ("day", "days"), "W": ("week", "weeks"), "M": ("month", "months"),
            "Q": ("quarter", "quarters"), "Y": ("year", "years")}

#: likelihood bins on the raw P(rise) (D.1.5; the narrative's ``_probability_line`` thresholds)
_LIKELIHOOD = (
    (lambda p: p >= 0.90, ("상승 가능성 매우 높음", "Rise very likely")),
    (lambda p: p >= 0.70, ("상승 쪽이 우세", "Rise more likely")),
    (lambda p: p >= 0.55, ("상승 쪽이 약간 우세", "Rise slightly more likely")),
    (lambda p: p > 0.45, ("상승·하락 엇비슷", "About even")),
    (lambda p: p > 0.30, ("하락 쪽이 약간 우세", "Fall slightly more likely")),
    (lambda p: p > 0.10, ("하락 쪽이 우세", "Fall more likely")),
    (lambda p: True, ("하락 가능성 매우 높음", "Fall very likely")),
)


def _en(lang: Any) -> bool:
    return lang == "en"


def _finite(x: Any) -> Optional[float]:
    """``float(x)`` for a finite real number (bools excluded), else ``None``."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------------------------------------------------

def _quantize(d: Decimal, places: int) -> Decimal:
    """Round an exact decimal to ``places`` decimals, ties away from zero."""
    return d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def round_half_away(x: float, d: int) -> Decimal:
    """The canonical display rounding (D.1): ``Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)``.

    ``Decimal(float)`` is exact, so binary ties such as ``0.125`` -> ``0.13`` and ``1234.5`` -> ``1235`` round away
    from zero. ``str()`` of the result is the reference string the JS ``roundHalfAway`` returns (e.g. ``"0.13"``,
    ``"1235"``, ``"-2.50"``; ``tests/js/fixtures/round_vectors.json``). Finite ``x`` only."""
    return _quantize(Decimal(x), int(d))


def _grouped(d: Decimal, places: int) -> str:
    """Digits of an already rounded non-negative decimal with thousands separators and exactly ``places``
    decimals (``Decimal`` keeps its exponent, so ``2.50`` stays ``2.50``)."""
    return f"{d:,.{places}f}"


def _sci_text(a: Decimal, digits: int = 3) -> str:
    """``×10ⁿ`` notation of the magnitude ``a`` (> 0) with ``digits`` significant digits, trailing zeros stripped:
    ``5×10⁻⁵``, ``3.4×10¹⁶``. No sign (the caller adds it)."""
    e = a.adjusted()
    mant = _quantize(a.scaleb(-e), digits - 1)
    if mant >= 10:                      # 9.995 -> 10.0: one more power of ten
        e += 1
        mant = _quantize(a.scaleb(-e), digits - 1)
    body = f"{mant:f}"
    if "." in body:
        body = body.rstrip("0").rstrip(".")
    return body if e == 0 else f"{body}×10{str(e).translate(_SUPERSCRIPT)}"


def _signed(negative: bool, body: str, plus: bool = False) -> str:
    if negative:
        return MINUS + body
    return ("+" + body) if plus else body


def _full_abs(a: Decimal) -> str:
    """:func:`full` of a non-negative exact decimal (no sign)."""
    if a == 0:
        return "0"
    if a < _SCI_SMALL or a >= _SCI_LARGE:
        return _sci_text(a)
    if a >= 1000:
        places = 0
    elif a >= 10:
        places = 1
    elif a >= Decimal("0.1"):
        places = 2
    else:                                # two significant digits, trailing zeros stripped
        places = -a.adjusted() + 1
        body = f"{_quantize(a, places):f}"
        return body.rstrip("0").rstrip(".") if "." in body else body
    return _grouped(_quantize(a, places), places)


# ---------------------------------------------------------------------------------------------------------------------
# Numbers (D.1.1 - D.1.3)
# ---------------------------------------------------------------------------------------------------------------------

def full(x: Optional[float], lang: str = "ko") -> str:
    """Full number for tables, tooltips and anomaly texts (D.1.2): ``number_text`` decimal rules (|x| >= 1,000 -> 0,
    >= 10 -> 1, >= 0.1 -> 2 decimals, 0 -> ``0``, smaller -> 2 significant digits, trailing zeros stripped; ×10ⁿ
    below 1e-4 and from 1e12) with :func:`round_half_away` and thousands separators. ``lang`` is accepted for
    symmetry (both languages print the same digits). ``None`` -> ``—``."""
    v = _finite(x)
    if v is None:
        return DASH
    body = _full_abs(abs(Decimal(v)))
    return _signed(v < 0 and body != "0", body)


def compact(x: Optional[float], lang: str = "ko") -> str:
    """Compact number for tiles, scenario boxes and axes (D.1.1): ``full(x)`` below 10,000 (after rounding); ko units
    만/억/조, en K (from 1e4)/M/B; at most 3 significant digits; ×10ⁿ from 1e16 (ko) / 1e12 (en).
    ``12345`` -> ``1.2만`` / ``12K``; ``9999.5`` -> ``1만`` / ``10K``; ``None`` -> ``—``."""
    v = _finite(x)
    if v is None:
        return DASH
    key = "en" if _en(lang) else "ko"
    a = abs(Decimal(v))
    if _quantize(a, 0) < 10000:
        return full(v, lang)
    neg = v < 0
    if a >= _COMPACT_SCI[key]:
        return _signed(neg, _sci_text(a))
    units, limit = _UNITS[key], _UNIT_LIMIT[key]
    i = 0
    while i + 1 < len(units) and a >= Decimal(1).scaleb(units[i + 1][0]):
        i += 1
    while i < len(units):
        power, suffix = units[i]
        u = a.scaleb(-power)
        r1 = _quantize(u, 1)
        if r1 < 10:
            body = f"{r1:f}"
            body = body[:-2] if body.endswith(".0") else body
            return _signed(neg, body + suffix)
        r0 = _quantize(u, 0)
        if r0 < 1000:
            return _signed(neg, f"{r0:f}" + suffix)
        if key == "ko":                  # 1,000만–9,999만 (and 억, 조): three significant digits
            r3 = _quantize(u.scaleb(-1), 0).scaleb(1)
            if r3 < limit:
                return _signed(neg, _grouped(r3, 0) + suffix)
        i += 1                           # the rounding reached the next unit
    return _signed(neg, _sci_text(a))


def value_text(x: Optional[float], kind: str, lang: str = "ko", compact: bool = False) -> str:
    """A data value by value kind (D.1.3): ``relative_index`` / ``relative_index_gt`` -> one fixed decimal
    (``71.3``, ``100.0``), never compacted; other kinds -> :func:`compact` when ``compact`` else :func:`full`."""
    v = _finite(x)
    if v is None:
        return DASH
    if kind in _INDEX_KINDS:
        r = _quantize(abs(Decimal(v)), 1)
        return _signed(v < 0 and r != 0, _grouped(r, 1))
    return _compact(v, lang) if compact else full(v, lang)


_compact = compact   # value_text's ``compact`` parameter shadows the function name


# ---------------------------------------------------------------------------------------------------------------------
# Percentages (D.1.4)
# ---------------------------------------------------------------------------------------------------------------------

def _pct_digits(v: float) -> Optional[str]:
    """Digits of a percentage (no sign, no ``%``); ``None`` when it rounds to 0 at one decimal."""
    a = abs(Decimal(v))
    r1 = _quantize(a, 1)
    if r1 == 0:
        return None
    if r1 >= 10:
        return _grouped(_quantize(a, 0), 0)
    return _grouped(r1, 1)


def pct(p: Optional[float]) -> str:
    """Signed percentage: 0 decimals when ``round1(|p|) >= 10`` else 1; ``+18%``, ``+4.1%``, ``−4.1%``,
    ``+1,365%``; ``round1(|p|) == 0`` -> ``0%``. ``None`` -> ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    digits = _pct_digits(v)
    if digits is None:
        return "0%"
    return _signed(v < 0, digits, plus=True) + "%"


def pct_arrow(p: Optional[float]) -> str:
    """Like :func:`pct` with ``▲``/``▼`` instead of the sign: ``▲12%``, ``▼4.1%``; zero -> ``0%``. ``None`` ->
    ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    digits = _pct_digits(v)
    if digits is None:
        return "0%"
    return ("▼" if v < 0 else "▲") + digits + "%"


def pct_abs(p: Optional[float]) -> str:
    """Digits of :func:`pct` without any sign (``18%``), for sentences that say 높을/낮을. ``None`` -> ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    digits = _pct_digits(v)
    return (digits or "0") + "%"


def pp(p: Optional[float], lang: str = "ko") -> str:
    """Signed percentage points: one fixed decimal when ``round1(|p|) >= 0.1`` (thousands separators from 1,000),
    ``0`` when ``p == 0``, else 2 significant digits with trailing zeros stripped (×10ⁿ below 1e-4, the notation
    rule of :func:`full`); suffix ``%p`` (ko) / `` pp`` (en): ``+0.4%p``, ``−3.9%p``, ``+0.042 pp``,
    ``+7.4×10⁻¹⁵%p``. ``None`` -> ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    suffix = " pp" if _en(lang) else "%p"
    if v == 0:
        return "0" + suffix
    a = abs(Decimal(v))
    r1 = _quantize(a, 1)
    if r1 >= Decimal("0.1"):
        body = _grouped(r1, 1)
    elif a < _SCI_SMALL:                 # float noise such as 7.4e-15: the ×10ⁿ rule of full(), 2 digits
        body = _sci_text(a, 2)
    else:
        places = -a.adjusted() + 1
        body = f"{_quantize(a, places):f}"
        body = body.rstrip("0").rstrip(".") if "." in body else body
    return _signed(v < 0, body, plus=True) + suffix


# ---------------------------------------------------------------------------------------------------------------------
# Probabilities (D.1.5)
# ---------------------------------------------------------------------------------------------------------------------

def prob_text(p: Optional[float], lang: str = "ko") -> str:
    """``p >= 0.975`` -> ``95% 이상`` / ``95% or more``; ``p <= 0.025`` -> ``5% 이하`` / ``5% or less``; else
    ``약 {round0(p·20)·5}%`` / ``about {…}%``. ``None`` -> ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    en = _en(lang)
    if v >= 0.975:
        return "95% or more" if en else "95% 이상"
    if v <= 0.025:
        return "5% or less" if en else "5% 이하"
    n = int(round_half_away(v * 20.0, 0)) * 5          # p·20 is a double product, as in fmt.js
    return f"about {n}%" if en else f"약 {n}%"


def prob_exact(p: Optional[float], lang: str = "ko") -> str:
    """``narrative.format_prob(p, lang)`` (``81%``, ``>99%``, ``<1%``) -- tooltips only. ``None`` -> ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    from ..narrative import format_prob   # lazy: keeps ``import trendai.web.fmt`` light

    return format_prob(v, "en" if _en(lang) else "ko")


def prob_labelled(p: Optional[float], lang: str = "ko") -> str:
    """``상승 확률 {prob_text}`` / ``chance of rise {prob_text}`` -- wherever the number stands next to a likelihood
    word. ``None`` -> ``—``."""
    if _finite(p) is None:
        return DASH
    text = prob_text(p, lang)
    return f"chance of rise {text}" if _en(lang) else f"상승 확률 {text}"


def likelihood(p: float, lang: str = "ko") -> str:
    """Likelihood word on the **raw** P(rise), the narrative's bins (D.1.5 table): ``>= 0.90`` 상승 가능성 매우 높음 /
    Rise very likely … ``<= 0.10`` 하락 가능성 매우 높음 / Fall very likely. ``None``/NaN -> ``—``."""
    v = _finite(p)
    if v is None:
        return DASH
    for test, words in _LIKELIHOOD:
        if test(v):
            return words[1] if _en(lang) else words[0]
    return DASH  # pragma: no cover - the last bin always matches


# ---------------------------------------------------------------------------------------------------------------------
# Dates and spans (D.1.6). Inputs are ``YYYY-MM-DD`` strings; timestamps are UTC ISO with ``Z``.
# ---------------------------------------------------------------------------------------------------------------------

def _date(iso: Any) -> _dt.date:
    """``YYYY-MM-DD`` (a longer ISO string or a date-like object is cut to its date part)."""
    if isinstance(iso, _dt.datetime):
        return iso.date()
    if isinstance(iso, _dt.date):
        return iso
    s = str(iso).strip()
    return _dt.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))


def _ko_day(d: _dt.date) -> str:
    return f"{d.year:04d}.{d.month:02d}.{d.day:02d}"


def _en_day(d: _dt.date) -> str:
    return f"{MONTHS_EN[d.month - 1]} {d.day}, {d.year}"


def _quarter(d: _dt.date) -> int:
    return (d.month - 1) // 3 + 1


def date_label(iso: str, freq: str, lang: str = "ko") -> str:
    """Long date label: D ``2026.11.25.`` / ``Nov 25, 2026``; W ``2026.11.17 주`` / ``Week of Nov 17, 2026``;
    M ``2026년 11월`` / ``Nov 2026``; Q ``2026년 4분기`` / ``Q4 2026``; Y ``2026년`` / ``2026``."""
    d = _date(iso)
    if _en(lang):
        if freq == "W":
            return f"Week of {_en_day(d)}"
        if freq == "M":
            return f"{MONTHS_EN[d.month - 1]} {d.year}"
        if freq == "Q":
            return f"Q{_quarter(d)} {d.year}"
        if freq == "Y":
            return f"{d.year}"
        return _en_day(d)
    if freq == "W":
        return f"{_ko_day(d)} 주"
    if freq == "M":
        return f"{d.year}년 {d.month}월"
    if freq == "Q":
        return f"{d.year}년 {_quarter(d)}분기"
    if freq == "Y":
        return f"{d.year}년"
    return f"{_ko_day(d)}."


def date_short(iso: str, freq: str, lang: str = "ko") -> str:
    """Short date for lists and narrative dates: D/W ``2026.11.25`` / ``Nov 25, 2026``; M ``2026.11`` /
    ``Nov 2026``; Q ``2026 4분기`` / ``Q4 2026``; Y ``2026``."""
    d = _date(iso)
    if _en(lang):
        if freq == "M":
            return f"{MONTHS_EN[d.month - 1]} {d.year}"
        if freq == "Q":
            return f"Q{_quarter(d)} {d.year}"
        if freq == "Y":
            return f"{d.year}"
        return _en_day(d)
    if freq == "M":
        return f"{d.year:04d}.{d.month:02d}"
    if freq == "Q":
        return f"{d.year} {_quarter(d)}분기"
    if freq == "Y":
        return f"{d.year}"
    return _ko_day(d)


def range_label(start: str, end: str, freq: str, lang: str = "ko") -> str:
    """Period label ``{a} ~ {b}`` (ko) / ``{a} – {b}`` (en) with the D.1.6 per-frequency forms. For ``W`` the end
    printed is ``end + 6 days`` (the last bucket's Sunday): ``range_label("2021-11-29", "2026-11-16", "W")`` ->
    ``2021.11.29 ~ 2026.11.22``."""
    a, b = _date(start), _date(end)
    if freq == "W":
        b = b + _dt.timedelta(days=6)
    if _en(lang):
        return f"{date_short(a, freq, 'en')} – {date_short(b, freq, 'en')}"
    if freq == "Q":   # 2021년 1분기 ~ 2026년 3분기 (the long form, D.1.6)
        return f"{date_label(a, 'Q', 'ko')} ~ {date_label(b, 'Q', 'ko')}"
    return f"{date_short(a, freq, 'ko')} ~ {date_short(b, freq, 'ko')}"


def datetime_label(utc_iso: str, lang: str = "ko") -> str:
    """A UTC timestamp shown in fixed KST (UTC+9, no tzdata): ``2026.11.25 09:12`` / ``Nov 25, 2026 09:12 KST``."""
    s = str(utc_iso).strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    t = _dt.datetime.fromisoformat(s)
    if t.tzinfo is not None:
        t = t.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    k = t + _KST
    hm = f"{k.hour:02d}:{k.minute:02d}"
    if _en(lang):
        return f"{_en_day(k.date())} {hm} KST"
    return f"{_ko_day(k.date())} {hm}"


def span_label(n: int, freq: str, lang: str = "ko") -> str:
    """Span words: ko ``26주``, ``12개월``, ``30일``, ``8분기``, ``5년``; en ``26 weeks``, ``12 months``, ``1 week``
    (singular for 1). Used for horizons, ``horizon_clamped``'s ``span`` and the tile-help ``{span}``."""
    k = int(n)
    if _en(lang):
        one, many = _SPAN_EN.get(freq, ("period", "periods"))
        return f"{k:,} {one if k == 1 else many}"
    return f"{k:,}{_SPAN_KO.get(freq, '기간')}"


def freq_label(freq: str, lang: str = "ko") -> str:
    """Frequency label: ko ``일 단위``/``주 단위``/``월 단위``/``분기 단위``/``연 단위``; en ``daily``/``weekly``/
    ``monthly``/``quarterly``/``yearly``."""
    ko, en = _FREQ_LABELS.get(freq, (freq, freq))
    return en if _en(lang) else ko


def freq_adjective(freq: str, lang: str = "ko") -> str:
    """Adjective for the summary line and chart desc: ko ``일간``/``주간``/``월간``/``분기``/``연간``; en the
    :func:`freq_label` words."""
    if _en(lang):
        return freq_label(freq, "en")
    return _FREQ_ADJ_KO.get(freq, freq)
