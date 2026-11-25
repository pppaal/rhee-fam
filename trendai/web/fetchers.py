"""Outbound data sources of the web app (spec B.3.2, E.2): Naver, Wikipedia and the LLM, injectable for tests."""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from .config import WebConfig

__all__ = ["Fetchers", "default_fetchers", "ai_status"]

LOG = logging.getLogger("trendai.web")


@dataclass
class Fetchers:
    """The callables the analyze and narrative jobs use. Attributes are read at call time, so tests may swap them."""

    naver: Optional[Callable[..., Any]]        # (groups, start, end, time_unit, device, gender, ages)
                                               # -> NaverResult-like; None = Naver not configured
    naver_endpoint: Optional[str]              # 'hub' | 'legacy' | None (shown in /api/config)
    wiki: Callable[..., pd.Series]             # (title, lang, start, end, granularity) -> pd.Series
    llm: Optional[Callable[[dict, str], str]]  # (llm_payload, lang) -> markdown; None = AI unavailable


def _anthropic_importable() -> bool:
    try:
        return importlib.util.find_spec("anthropic") is not None
    except (ImportError, ValueError):
        return False


def ai_status(config: WebConfig, environ: Mapping[str, str] = os.environ) -> tuple[bool, Optional[str]]:
    """``(on, reason)`` of the AI feature for the default fetchers: ``TRENDAI_AI=off`` -> ``(False, "disabled")``;
    ``auto`` -> on iff ``anthropic`` is importable and ``ANTHROPIC_API_KEY`` is set (else ``"unconfigured"``);
    ``on`` -> on iff ``anthropic`` is importable."""
    if config.ai == "off":
        return False, "disabled"
    if not _anthropic_importable():
        return False, "unconfigured"
    if config.ai == "auto" and not str(environ.get("ANTHROPIC_API_KEY") or "").strip():
        return False, "unconfigured"
    return True, None


def _wiki(fetch: Callable[..., pd.Series], session: Any, user_agent: str,
          title: str, lang: str, start: Any, end: Any, granularity: str) -> pd.Series:
    return fetch(title, lang=lang, start=start, end=end, granularity=granularity, session=session,
                 user_agent=user_agent)


def default_fetchers(config: WebConfig) -> Fetchers:
    """The production fetchers (E.2): Naver from ``NaverCredentials.from_env()`` (``None`` when unset), Wikipedia with
    the contact User-Agent, and Claude when AI is on (one client with a 90 s timeout and one retry, C.11). One
    stateless ``UrllibSession`` is shared by all of them."""
    from .. import connectors

    session = connectors.UrllibSession()
    creds = connectors.NaverCredentials.from_env()
    naver = None
    endpoint = None
    if creds is not None:
        naver = functools.partial(connectors.fetch_naver_search_trend, credentials=creds, session=session)
        endpoint = creds.endpoint
    ua = connectors.user_agent(config.public_url, config.contact)
    wiki = functools.partial(_wiki, connectors.fetch_wikipedia_pageviews, session, ua)
    llm = None
    on, _reason = ai_status(config)
    if on:
        try:
            import anthropic

            from .. import narrative

            client = anthropic.Anthropic(timeout=90.0, max_retries=1)
            model = config.ai_model

            def llm(payload: dict, lang: str) -> str:
                return narrative.llm_narrative_from_payload(payload, lang, model=model, client=client)
        except Exception as exc:  # noqa: BLE001 - AI is optional; the app runs without it
            LOG.warning("AI disabled: the Anthropic client could not be created (%s)", type(exc).__name__)
            llm = None
    return Fetchers(naver=naver, naver_endpoint=endpoint, wiki=wiki, llm=llm)
