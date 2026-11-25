"""Web app configuration (spec E.1): one frozen :class:`WebConfig` read from ``TRENDAI_*`` variables.

API keys are never stored here: the Naver keys are read only by ``connectors.NaverCredentials.from_env`` and
``ANTHROPIC_API_KEY`` only by the Anthropic SDK (``fetchers.default_fetchers`` merely checks that it is set), so
``repr(WebConfig)`` can never show a secret.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

__all__ = ["WebConfig", "parse_bool"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
_AI_MODES = ("auto", "on", "off")
#: Naver data may be cached at most this many days (A.5: the 2026-09 terms)
_NAVER_CACHE_MAX_DAYS = 21


def parse_bool(value: Any, name: str) -> bool:
    """``1/true/yes/on`` and ``0/false/no/off`` (case-insensitive); a real bool passes through."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{name}: expected a boolean (1/0, true/false, yes/no, on/off), got {value!r}")


def _int(value: Any, name: str, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    try:
        number = int(str(value).strip()) if not isinstance(value, int) else value
    except ValueError:
        raise ValueError(f"{name}: expected an integer, got {value!r}") from None
    if number < minimum or (maximum is not None and number > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name}: must be >= {minimum}{upper}, got {number}")
    return number


def _float(value: Any, name: str, *, positive: bool = True, maximum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a number, got {value!r}")
    try:
        number = float(str(value).strip()) if not isinstance(value, (int, float)) else float(value)
    except ValueError:
        raise ValueError(f"{name}: expected a number, got {value!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name}: expected a finite number, got {value!r}")
    if positive and number <= 0:
        raise ValueError(f"{name}: must be > 0, got {number}")
    if not positive and number < 0:
        raise ValueError(f"{name}: must be >= 0, got {number}")
    if maximum is not None and number > maximum:
        # an upper bound where a huge value would only fail later (a TTL of 1e9 hours overflows the date arithmetic
        # of every analysis and share): refused at start-up, naming the variable (E.1)
        raise ValueError(f"{name}: must be <= {maximum:g}, got {number:g}")
    return number


def _opt_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name}: must not be empty")
    return text


def _log_level(value: Any, name: str) -> str:
    text = str(value).strip().upper()
    if text not in _LOG_LEVELS:
        raise ValueError(f"{name}: expected one of {', '.join(_LOG_LEVELS)}, got {value!r}")
    return text


def _ai_mode(value: Any, name: str) -> str:
    text = str(value).strip().lower()
    if text not in _AI_MODES:
        raise ValueError(f"{name}: expected auto, on or off, got {value!r}")
    return text


def _public_url(value: Any, name: str) -> Optional[str]:
    text = _opt_str(value, name)
    if text is None:
        return None
    if not (text.startswith("http://") or text.startswith("https://")) or "://" not in text[:9]:
        raise ValueError(f"{name}: expected an http(s) URL such as https://trendai.example, got {value!r}")
    return text.rstrip("/")


def _cache_days(value: Any, name: str) -> int:
    return min(_int(value, name, minimum=1), _NAVER_CACHE_MAX_DAYS)


#: field -> (environment variable or None, parser). The order is the E.1 table's.
_FIELDS: dict[str, tuple[Optional[str], Callable[[Any, str], Any]]] = {
    "host": ("TRENDAI_HOST", _str),
    "port": ("TRENDAI_PORT", lambda v, n: _int(v, n, minimum=0, maximum=65535)),
    "data_dir": ("TRENDAI_DATA_DIR", _str),
    "public_url": ("TRENDAI_PUBLIC_URL", _public_url),
    "contact": ("TRENDAI_CONTACT", _opt_str),
    "allow_no_contact": ("TRENDAI_ALLOW_NO_CONTACT", parse_bool),
    "trust_proxy": ("TRENDAI_TRUST_PROXY", parse_bool),
    "static_dir": (None, _opt_str),
    "dev": ("TRENDAI_DEV", parse_bool),
    "log_level": ("TRENDAI_LOG_LEVEL", _log_level),
    "workers": ("TRENDAI_WORKERS", lambda v, n: _int(v, n, minimum=1, maximum=64)),
    "max_queued": ("TRENDAI_MAX_QUEUED", lambda v, n: _int(v, n, minimum=0, maximum=10_000)),
    "job_timeout_s": ("TRENDAI_JOB_TIMEOUT", lambda v, n: _float(v, n, maximum=86_400)),
    "upload_max_mb": ("TRENDAI_UPLOAD_MAX_MB", lambda v, n: _int(v, n, minimum=1, maximum=1024)),
    "upload_max_series": ("TRENDAI_UPLOAD_MAX_SERIES", lambda v, n: _int(v, n, minimum=1, maximum=99)),
    "result_ttl_h": ("TRENDAI_RESULT_TTL_H", lambda v, n: _float(v, n, maximum=8_760)),       # one year
    "share_ttl_d": ("TRENDAI_SHARE_TTL_D", lambda v, n: _float(v, n, maximum=3_650)),        # ten years
    "naver_monthly_limit": ("TRENDAI_NAVER_MONTHLY_LIMIT", lambda v, n: _int(v, n, minimum=1)),
    "naver_ip_per_hour": ("TRENDAI_NAVER_IP_PER_HOUR", lambda v, n: _int(v, n, minimum=1)),
    "naver_cache_days": ("TRENDAI_NAVER_CACHE_DAYS", _cache_days),
    "ai": ("TRENDAI_AI", _ai_mode),
    "ai_model": ("TRENDAI_AI_MODEL", _str),
    "ai_ip_per_hour": ("TRENDAI_AI_IP_PER_HOUR", lambda v, n: _int(v, n, minimum=1)),
    "ai_daily_limit": ("TRENDAI_AI_DAILY_LIMIT", lambda v, n: _int(v, n, minimum=0)),
    "hsts": ("TRENDAI_HSTS", parse_bool),
    "store_max_mb": ("TRENDAI_STORE_MAX_MB", lambda v, n: _int(v, n, minimum=1)),
    "shutdown_grace_s": ("TRENDAI_SHUTDOWN_GRACE", lambda v, n: _float(v, n, positive=False, maximum=3_600)),
}


@dataclass(frozen=True)
class WebConfig:
    """Every setting of the web app (E.1). Defaults are the E.1 defaults; build it with :meth:`from_env`."""

    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: str = "./.trendai-data"
    public_url: Optional[str] = None
    contact: Optional[str] = None
    allow_no_contact: bool = False
    trust_proxy: bool = False
    static_dir: Optional[str] = None     # override only (no env var): None -> the package's static/ directory
    dev: bool = False
    log_level: str = "INFO"
    workers: int = 2
    max_queued: int = 8
    job_timeout_s: float = 180.0
    upload_max_mb: int = 10
    upload_max_series: int = 20
    result_ttl_h: float = 24.0
    share_ttl_d: float = 30.0
    naver_monthly_limit: int = 50_000
    naver_ip_per_hour: int = 20
    naver_cache_days: int = 21
    ai: str = "auto"
    ai_model: str = "claude-opus-5-5"
    ai_ip_per_hour: int = 10
    ai_daily_limit: int = 200
    hsts: bool = False
    store_max_mb: int = 2048
    shutdown_grace_s: float = 30.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ, **overrides: Any) -> "WebConfig":
        """Read every ``TRENDAI_*`` variable of E.1 from ``environ``, then apply ``overrides`` (field names; values
        may be strings, as from a command line, or typed). Invalid values raise ``ValueError`` naming the variable
        (or the field for an override); an unknown override raises ``TypeError``."""
        values: dict[str, Any] = {}
        for field_name, (env_name, parser) in _FIELDS.items():
            if env_name is None:
                continue
            raw = environ.get(env_name)
            if raw is None or str(raw).strip() == "":
                continue
            values[field_name] = parser(raw, env_name)
        for key, raw in overrides.items():
            if key not in _FIELDS:
                raise TypeError(f"WebConfig has no setting {key!r}")
            if raw is None:
                if key in ("public_url", "contact", "static_dir"):
                    values[key] = None
                continue
            if key in ("data_dir", "static_dir") and isinstance(raw, os.PathLike):
                raw = os.fspath(raw)
            values[key] = _FIELDS[key][1](raw, key)
        return cls(**values)

    def replace(self, **changes: Any) -> "WebConfig":
        """A copy with some fields changed (validated like :meth:`from_env` overrides)."""
        checked = {k: _FIELDS[k][1](v, k) if v is not None else None for k, v in changes.items()}
        return dataclasses.replace(self, **checked)

    def is_public(self) -> bool:
        """True when the bind address is not loopback (127.0.0.0/8, ::1, ``localhost``) or ``public_url`` is set."""
        if self.public_url:
            return True
        host = self.host.strip().strip("[]").lower()
        if host == "localhost" or host.endswith(".localhost"):
            return False
        try:
            return not ipaddress.ip_address(host).is_loopback
        except ValueError:
            return True   # a host name other than localhost: assume it is reachable from outside

    def validate_public(self) -> None:
        """``ValueError`` when the app is public (:meth:`is_public`) without ``TRENDAI_CONTACT`` and without
        ``TRENDAI_ALLOW_NO_CONTACT`` (A.5, E.1). Called by ``trendai serve`` and ``trendai.web.wsgi``."""
        if self.is_public() and not self.contact and not self.allow_no_contact:
            raise ValueError("TRENDAI_CONTACT must be set (operator e-mail or URL) to run on a public address; "
                             "set TRENDAI_ALLOW_NO_CONTACT=1 for private use")

    @property
    def upload_max_bytes(self) -> int:
        """The upload size limit in bytes (``upload_max_mb`` x 1024 x 1024, B.3.4)."""
        return int(self.upload_max_mb) * 1024 * 1024

    @property
    def result_ttl_s(self) -> float:
        return float(self.result_ttl_h) * 3600.0

    @property
    def share_ttl_s(self) -> float:
        return float(self.share_ttl_d) * 86400.0
