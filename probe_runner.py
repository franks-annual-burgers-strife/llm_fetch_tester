from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from providers import (
    APIRequestSpec,
    PRESETS,
    ProviderApiConfig,
    build_claude_access_request,
    build_gemini_access_request,
    build_openai_diagnostic_request,
)


SECRET_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
}

URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+")
BLOCK_PAGE_HINTS = (
    "access denied",
    "attention required",
    "blocked",
    "bot detection",
    "captcha",
    "challenge",
    "cloudflare",
    "forbidden",
    "login required",
    "paywall",
    "please verify you are human",
    "security check",
    "sorry, you have been blocked",
    "verify you are human",
)


@dataclass(frozen=True)
class HttpCallResult:
    provider_id: str
    provider_name: str
    request_kind: str
    method: str
    url: str
    model: str
    started_at: str
    completed_at: str
    duration_ms: int
    request_headers: dict[str, str]
    request_json: dict[str, Any] | None
    status_code: int | None
    final_url: str | None
    response_headers: dict[str, str]
    raw_body: str
    parsed_json: Any | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    label: str
    value: str
    url: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    control_available: bool
    matches_requested_url: bool
    matched_url: str | None
    title_match: bool | None
    heading_match: bool | None
    quote_hits: int
    plausible_content: bool
    confidence: float
    mismatch_notes: list[str] = field(default_factory=list)
    block_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlFingerprintResult:
    requested_url: str
    verdict: str
    summary: str
    started_at: str
    completed_at: str
    duration_ms: int
    status_code: int | None
    final_url: str | None
    content_type: str
    page_title: str | None
    main_heading: str | None
    canonical_url: str | None
    snippet: str
    raw_excerpt: str
    response_headers: dict[str, str]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderAccessResult:
    provider_id: str
    provider_name: str
    mode_label: str
    model: str
    exact_url_supported: bool
    requested_url: str
    verdict: str
    confidence: float
    summary: str
    evidence_url: str | None
    extracted_title: str | None
    extracted_heading: str | None
    quotes: list[str]
    facts: list[str]
    blocker_reason: str | None
    evidence_records: list[EvidenceRecord]
    citations: list[EvidenceRecord]
    tool_metadata: dict[str, Any]
    verification: VerificationResult
    secondary_verdict: str | None
    secondary_summary: str | None
    request_headers: dict[str, str]
    request_json: dict[str, Any] | None
    response_status: int | None
    response_headers: dict[str, str]
    raw_response_json: Any | None
    raw_response_text: str
    started_at: str
    completed_at: str
    duration_ms: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetUrlRunConfig:
    target_url: str
    timeout_seconds: float
    openai: ProviderApiConfig
    gemini: ProviderApiConfig
    anthropic: ProviderApiConfig
    run_openai_diagnostic: bool = True


@dataclass(frozen=True)
class TargetUrlRunResult:
    generated_at: str
    target_url: str
    control: ControlFingerprintResult
    providers: dict[str, ProviderAccessResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _HTMLFingerprintParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: str | None = None
        self.main_heading: str | None = None
        self.canonical_url: str | None = None
        self._in_title = False
        self._in_h1 = False
        self._skip_depth = 0
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs if key}
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag == "title" and self.title is None:
            self._in_title = True
        if tag == "h1" and self.main_heading is None:
            self._in_h1 = True
        if tag == "link" and self.canonical_url is None:
            rel_value = (attrs_dict.get("rel") or "").lower()
            if "canonical" in rel_value:
                self.canonical_url = attrs_dict.get("href")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag == "h1":
            self._in_h1 = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = _normalize_text(data)
        if not cleaned:
            return
        if self._in_title and self.title is None:
            self.title = cleaned
        if self._in_h1 and self.main_heading is None:
            self.main_heading = cleaned
        self._text_chunks.append(cleaned)

    def build_snippet(self, limit: int = 280) -> str:
        merged = _normalize_text(" ".join(self._text_chunks))
        return trim_text(merged, limit=limit)


def normalize_target_url(raw_url: str) -> str:
    value = raw_url.strip()
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.scheme:
        return f"https://{value}"
    return value


def run_url_access_suite(config: TargetUrlRunConfig, client: httpx.Client | None = None) -> TargetUrlRunResult:
    target_url = normalize_target_url(config.target_url)
    control = fetch_control_fingerprint(target_url, timeout_seconds=config.timeout_seconds, client=client)
    providers = {
        "gemini": run_gemini_access_check(config.gemini, target_url, control, client=client),
        "anthropic": run_claude_access_check(config.anthropic, target_url, control, client=client),
        "openai": run_openai_access_check(
            config.openai,
            target_url,
            control,
            run_diagnostic=config.run_openai_diagnostic,
            client=client,
        ),
    }
    return TargetUrlRunResult(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        target_url=target_url,
        control=control,
        providers=providers,
    )


def fetch_control_fingerprint(
    target_url: str,
    timeout_seconds: float = 20.0,
    client: httpx.Client | None = None,
) -> ControlFingerprintResult:
    started_at = datetime.now(timezone.utc)
    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True, timeout=timeout_seconds)
    try:
        response = http_client.get(
            target_url,
            headers={
                "Accept": "text/html,application/pdf,application/json;q=0.9,*/*;q=0.8",
                "User-Agent": "llm-url-access-tester/1.0",
            },
        )
        completed_at = datetime.now(timezone.utc)
        content_type = response.headers.get("content-type", "")
        parser = _HTMLFingerprintParser()
        page_title = None
        main_heading = None
        canonical_url = None
        snippet = ""
        raw_excerpt = ""
        if _looks_like_html(content_type, response.text):
            parser.feed(response.text)
            page_title = parser.title
            main_heading = parser.main_heading
            canonical_url = parser.canonical_url
            snippet = parser.build_snippet()
            raw_excerpt = trim_text(response.text, limit=2500)
        elif "pdf" in content_type.lower():
            snippet = "PDF response detected. No HTML fingerprint was extracted."
        elif response.text.strip():
            snippet = trim_text(_normalize_text(response.text), limit=280)
            raw_excerpt = trim_text(response.text, limit=2500)
        summary = f"Validation-only local fingerprint fetched with HTTP {response.status_code}."
        return ControlFingerprintResult(
            requested_url=target_url,
            verdict="available",
            summary=summary,
            started_at=_format_ts(started_at),
            completed_at=_format_ts(completed_at),
            duration_ms=_duration_ms(started_at, completed_at),
            status_code=response.status_code,
            final_url=str(response.url),
            content_type=content_type,
            page_title=page_title,
            main_heading=main_heading,
            canonical_url=canonical_url,
            snippet=snippet,
            raw_excerpt=raw_excerpt,
            response_headers=dict(response.headers.items()),
            error=None,
        )
    except httpx.RequestError as exc:
        completed_at = datetime.now(timezone.utc)
        return ControlFingerprintResult(
            requested_url=target_url,
            verdict="unavailable",
            summary="Validation-only local fingerprint could not be fetched.",
            started_at=_format_ts(started_at),
            completed_at=_format_ts(completed_at),
            duration_ms=_duration_ms(started_at, completed_at),
            status_code=None,
            final_url=None,
            content_type="",
            page_title=None,
            main_heading=None,
            canonical_url=None,
            snippet="",
            raw_excerpt="",
            response_headers={},
            error=str(exc) or exc.__class__.__name__,
        )
    finally:
        if owns_client:
            http_client.close()


def run_gemini_access_check(
    config: ProviderApiConfig,
    target_url: str,
    control: ControlFingerprintResult,
    client: httpx.Client | None = None,
) -> ProviderAccessResult:
    preset = PRESETS["gemini"]
    if not config.api_key.strip():
        return _build_stub_result(
            provider_id="gemini",
            provider_name=preset.display_name,
            mode_label="Exact URL Context",
            model=config.model,
            target_url=target_url,
            exact_url_supported=True,
            verdict="auth_failed",
            confidence=0.2,
            summary="No Gemini API key supplied.",
        )

    request = build_gemini_access_request(target_url, config)
    http_result = execute_request(request, client=client)
    error_outcome = _http_error_outcome(http_result)
    if error_outcome is not None:
        verdict, summary = error_outcome
        return _build_result_from_http_only(
            http_result=http_result,
            target_url=target_url,
            mode_label="Exact URL Context",
            exact_url_supported=True,
            verdict=verdict,
            confidence=0.25 if verdict == "provider_error" else 0.2,
            summary=summary,
        )

    payload = http_result.parsed_json if isinstance(http_result.parsed_json, dict) else {}
    text = _extract_gemini_text(payload)
    extraction = _extract_structured_payload(text)
    metadata = _extract_gemini_metadata(payload)
    candidate_urls = _unique_urls(_find_urls_in_object(metadata) + [_nullable_str(extraction.get("observed_url"))])
    verification = _verify_access(
        requested_url=target_url,
        candidate_urls=candidate_urls,
        control=control,
        extracted_title=_nullable_str(extraction.get("page_title")),
        extracted_heading=_nullable_str(extraction.get("main_heading")),
        quotes=_normalize_string_list(extraction.get("quotes")),
        facts=_normalize_string_list(extraction.get("facts")),
        blocker_reason=_nullable_str(extraction.get("blocker_reason")),
    )
    verdict, summary, confidence = _classify_exact_access(
        provider_name=preset.display_name,
        verification=verification,
        blocker_reason=_nullable_str(extraction.get("blocker_reason")),
        extracted_title=_nullable_str(extraction.get("page_title")),
        extracted_heading=_nullable_str(extraction.get("main_heading")),
        quotes=_normalize_string_list(extraction.get("quotes")),
        facts=_normalize_string_list(extraction.get("facts")),
    )
    return ProviderAccessResult(
        provider_id="gemini",
        provider_name=preset.display_name,
        mode_label="Exact URL Context",
        model=config.model,
        exact_url_supported=True,
        requested_url=target_url,
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        evidence_url=verification.matched_url or _first_url(candidate_urls),
        extracted_title=_nullable_str(extraction.get("page_title")),
        extracted_heading=_nullable_str(extraction.get("main_heading")),
        quotes=_normalize_string_list(extraction.get("quotes")),
        facts=_normalize_string_list(extraction.get("facts")),
        blocker_reason=_nullable_str(extraction.get("blocker_reason")),
        evidence_records=_build_evidence_records(extraction),
        citations=_urls_to_evidence(candidate_urls),
        tool_metadata=metadata,
        verification=verification,
        secondary_verdict=None,
        secondary_summary=None,
        request_headers=http_result.request_headers,
        request_json=http_result.request_json,
        response_status=http_result.status_code,
        response_headers=http_result.response_headers,
        raw_response_json=payload,
        raw_response_text=text or http_result.raw_body,
        started_at=http_result.started_at,
        completed_at=http_result.completed_at,
        duration_ms=http_result.duration_ms,
        error=http_result.error,
    )


def run_claude_access_check(
    config: ProviderApiConfig,
    target_url: str,
    control: ControlFingerprintResult,
    client: httpx.Client | None = None,
) -> ProviderAccessResult:
    preset = PRESETS["anthropic"]
    if not config.api_key.strip():
        return _build_stub_result(
            provider_id="anthropic",
            provider_name=preset.display_name,
            mode_label="Exact Web Fetch",
            model=config.model,
            target_url=target_url,
            exact_url_supported=True,
            verdict="auth_failed",
            confidence=0.2,
            summary="No Anthropic API key supplied.",
        )

    request = build_claude_access_request(target_url, config)
    http_result = execute_request(request, client=client)
    error_outcome = _http_error_outcome(http_result)
    if error_outcome is not None:
        verdict, summary = error_outcome
        return _build_result_from_http_only(
            http_result=http_result,
            target_url=target_url,
            mode_label="Exact Web Fetch",
            exact_url_supported=True,
            verdict=verdict,
            confidence=0.25 if verdict == "provider_error" else 0.2,
            summary=summary,
        )

    payload = http_result.parsed_json if isinstance(http_result.parsed_json, dict) else {}
    text_blocks = _extract_claude_text_blocks(payload)
    extraction = _extract_structured_payload("\n".join(text_blocks))
    fetch_results = _extract_claude_fetch_results(payload)
    candidate_urls = _unique_urls(
        [item.get("url") for item in fetch_results] + _find_urls_in_object(fetch_results) + [_nullable_str(extraction.get("observed_url"))]
    )
    extracted_title = _nullable_str(extraction.get("page_title")) or _first_nonempty(
        *(item.get("title") for item in fetch_results)
    )
    extracted_heading = _nullable_str(extraction.get("main_heading"))
    quotes = _normalize_string_list(extraction.get("quotes"))
    facts = _normalize_string_list(extraction.get("facts"))
    blocker_reason = _nullable_str(extraction.get("blocker_reason"))
    verification = _verify_access(
        requested_url=target_url,
        candidate_urls=candidate_urls,
        control=control,
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
        blocker_reason=blocker_reason,
    )
    verdict, summary, confidence = _classify_exact_access(
        provider_name=preset.display_name,
        verification=verification,
        blocker_reason=blocker_reason,
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
    )
    return ProviderAccessResult(
        provider_id="anthropic",
        provider_name=preset.display_name,
        mode_label="Exact Web Fetch",
        model=config.model,
        exact_url_supported=True,
        requested_url=target_url,
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        evidence_url=verification.matched_url or _first_url(candidate_urls),
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
        blocker_reason=blocker_reason,
        evidence_records=_build_evidence_records(extraction),
        citations=_extract_citations_from_object(payload),
        tool_metadata={
            "tool_use_blocks": _extract_claude_tool_use_blocks(payload),
            "web_fetch_results": fetch_results,
        },
        verification=verification,
        secondary_verdict=None,
        secondary_summary=None,
        request_headers=http_result.request_headers,
        request_json=http_result.request_json,
        response_status=http_result.status_code,
        response_headers=http_result.response_headers,
        raw_response_json=payload,
        raw_response_text="\n".join(text_blocks) or http_result.raw_body,
        started_at=http_result.started_at,
        completed_at=http_result.completed_at,
        duration_ms=http_result.duration_ms,
        error=http_result.error,
    )


def run_openai_access_check(
    config: ProviderApiConfig,
    target_url: str,
    control: ControlFingerprintResult,
    run_diagnostic: bool = True,
    client: httpx.Client | None = None,
) -> ProviderAccessResult:
    preset = PRESETS["openai"]
    if not run_diagnostic:
        return _build_stub_result(
            provider_id="openai",
            provider_name=preset.display_name,
            mode_label="Search-Based Access Check",
            model=config.model,
            target_url=target_url,
            exact_url_supported=False,
            verdict="inconclusive",
            confidence=0.2,
            summary="OpenAI search-based access check was disabled.",
        )
    if not config.api_key.strip():
        return _build_stub_result(
            provider_id="openai",
            provider_name=preset.display_name,
            mode_label="Search-Based Access Check",
            model=config.model,
            target_url=target_url,
            exact_url_supported=False,
            verdict="auth_failed",
            confidence=0.2,
            summary="No OpenAI API key supplied.",
        )

    request = build_openai_diagnostic_request(target_url, config)
    http_result = execute_request(request, client=client)
    error_outcome = _http_error_outcome(http_result)
    if error_outcome is not None:
        verdict, summary = error_outcome
        return _build_result_from_http_only(
            http_result=http_result,
            target_url=target_url,
            mode_label="Search-Based Access Check",
            exact_url_supported=False,
            verdict=verdict,
            confidence=0.25 if verdict == "provider_error" else 0.2,
            summary=summary,
        )

    payload = http_result.parsed_json if isinstance(http_result.parsed_json, dict) else {}
    text = _extract_openai_output_text(payload)
    extraction = _extract_structured_payload(text)
    citations = _extract_openai_citations(payload)
    web_search_actions = _extract_openai_web_search_actions(payload)
    action_urls = _extract_openai_action_urls(web_search_actions)
    candidate_urls = _unique_urls(
        [item.url for item in citations] + action_urls + [_nullable_str(extraction.get("observed_url"))]
    )
    quotes = _normalize_string_list(extraction.get("quotes"))
    facts = _normalize_string_list(extraction.get("facts"))
    extracted_title = _nullable_str(extraction.get("page_title"))
    extracted_heading = _nullable_str(extraction.get("main_heading"))
    blocker_reason = _nullable_str(extraction.get("blocker_reason"))
    verification = _verify_access(
        requested_url=target_url,
        candidate_urls=candidate_urls,
        control=control,
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
        blocker_reason=blocker_reason,
    )
    diagnostic_verdict, diagnostic_summary, diagnostic_confidence = _classify_openai_diagnostic(
        payload=payload,
        target_url=target_url,
        verification=verification,
        blocker_reason=blocker_reason,
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
        actions=web_search_actions,
    )
    return ProviderAccessResult(
        provider_id="openai",
        provider_name=preset.display_name,
        mode_label="Search-Based Access Check",
        model=config.model,
        exact_url_supported=False,
        requested_url=target_url,
        verdict=diagnostic_verdict,
        confidence=diagnostic_confidence,
        summary=diagnostic_summary,
        evidence_url=verification.matched_url or _first_url(candidate_urls),
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
        blocker_reason=blocker_reason,
        evidence_records=_build_evidence_records(extraction),
        citations=citations,
        tool_metadata={"web_search_actions": web_search_actions},
        verification=verification,
        secondary_verdict=None,
        secondary_summary=None,
        request_headers=http_result.request_headers,
        request_json=http_result.request_json,
        response_status=http_result.status_code,
        response_headers=http_result.response_headers,
        raw_response_json=payload,
        raw_response_text=text or http_result.raw_body,
        started_at=http_result.started_at,
        completed_at=http_result.completed_at,
        duration_ms=http_result.duration_ms,
        error=http_result.error,
    )


def execute_request(request: APIRequestSpec, client: httpx.Client | None = None) -> HttpCallResult:
    started_at = datetime.now(timezone.utc)
    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True, timeout=request.timeout_seconds)
    try:
        response = http_client.request(
            request.method,
            request.url,
            headers=request.headers,
            json=request.json_body,
        )
        completed_at = datetime.now(timezone.utc)
        raw_body = response.text
        return HttpCallResult(
            provider_id=request.provider_id,
            provider_name=request.provider_name,
            request_kind=request.request_kind,
            method=request.method,
            url=request.url,
            model=request.model,
            started_at=_format_ts(started_at),
            completed_at=_format_ts(completed_at),
            duration_ms=_duration_ms(started_at, completed_at),
            request_headers=redact_headers(request.headers),
            request_json=request.json_body,
            status_code=response.status_code,
            final_url=str(response.url),
            response_headers=dict(response.headers.items()),
            raw_body=raw_body,
            parsed_json=_parse_json(raw_body, response.headers.get("content-type", "")),
            error=None,
        )
    except httpx.RequestError as exc:
        completed_at = datetime.now(timezone.utc)
        return HttpCallResult(
            provider_id=request.provider_id,
            provider_name=request.provider_name,
            request_kind=request.request_kind,
            method=request.method,
            url=request.url,
            model=request.model,
            started_at=_format_ts(started_at),
            completed_at=_format_ts(completed_at),
            duration_ms=_duration_ms(started_at, completed_at),
            request_headers=redact_headers(request.headers),
            request_json=request.json_body,
            status_code=None,
            final_url=None,
            response_headers={},
            raw_body="",
            parsed_json=None,
            error=str(exc) or exc.__class__.__name__,
        )
    finally:
        if owns_client:
            http_client.close()


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in SECRET_HEADER_NAMES and value:
            redacted[name] = _mask_secret(value)
        else:
            redacted[name] = value
    return redacted


def trim_text(value: str, limit: int = 1600) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n\n... [trimmed {len(value) - limit} characters]"


def detect_code_language(raw: Any) -> str:
    if isinstance(raw, (dict, list)):
        return "json"
    text = str(raw or "")
    lowered = text.lstrip().lower()
    if lowered.startswith("{") or lowered.startswith("["):
        return "json"
    if lowered.startswith("<!doctype html") or lowered.startswith("<html"):
        return "html"
    if lowered.startswith("<?xml"):
        return "xml"
    return "text"


def _build_stub_result(
    provider_id: str,
    provider_name: str,
    mode_label: str,
    model: str,
    target_url: str,
    exact_url_supported: bool,
    verdict: str,
    confidence: float,
    summary: str,
    secondary_verdict: str | None = None,
    secondary_summary: str | None = None,
) -> ProviderAccessResult:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    verification = VerificationResult(
        control_available=False,
        matches_requested_url=False,
        matched_url=None,
        title_match=None,
        heading_match=None,
        quote_hits=0,
        plausible_content=False,
        confidence=confidence,
        mismatch_notes=[],
        block_signals=[],
    )
    return ProviderAccessResult(
        provider_id=provider_id,
        provider_name=provider_name,
        mode_label=mode_label,
        model=model,
        exact_url_supported=exact_url_supported,
        requested_url=target_url,
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        evidence_url=None,
        extracted_title=None,
        extracted_heading=None,
        quotes=[],
        facts=[],
        blocker_reason=None,
        evidence_records=[],
        citations=[],
        tool_metadata={},
        verification=verification,
        secondary_verdict=secondary_verdict,
        secondary_summary=secondary_summary,
        request_headers={},
        request_json=None,
        response_status=None,
        response_headers={},
        raw_response_json=None,
        raw_response_text="",
        started_at=now,
        completed_at=now,
        duration_ms=0,
        error=None,
    )


def _build_result_from_http_only(
    http_result: HttpCallResult,
    target_url: str,
    mode_label: str,
    exact_url_supported: bool,
    verdict: str,
    confidence: float,
    summary: str,
    secondary_verdict: str | None = None,
    secondary_summary: str | None = None,
) -> ProviderAccessResult:
    verification = VerificationResult(
        control_available=False,
        matches_requested_url=False,
        matched_url=None,
        title_match=None,
        heading_match=None,
        quote_hits=0,
        plausible_content=False,
        confidence=confidence,
        mismatch_notes=[],
        block_signals=[],
    )
    return ProviderAccessResult(
        provider_id=http_result.provider_id,
        provider_name=http_result.provider_name,
        mode_label=mode_label,
        model=http_result.model,
        exact_url_supported=exact_url_supported,
        requested_url=target_url,
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        evidence_url=None,
        extracted_title=None,
        extracted_heading=None,
        quotes=[],
        facts=[],
        blocker_reason=None,
        evidence_records=[],
        citations=[],
        tool_metadata={},
        verification=verification,
        secondary_verdict=secondary_verdict,
        secondary_summary=secondary_summary,
        request_headers=http_result.request_headers,
        request_json=http_result.request_json,
        response_status=http_result.status_code,
        response_headers=http_result.response_headers,
        raw_response_json=http_result.parsed_json,
        raw_response_text=http_result.raw_body,
        started_at=http_result.started_at,
        completed_at=http_result.completed_at,
        duration_ms=http_result.duration_ms,
        error=http_result.error,
    )


def _http_error_outcome(http_result: HttpCallResult) -> tuple[str, str] | None:
    if http_result.error:
        return "provider_error", f"Request failed before the provider returned a response: {http_result.error}"
    error_message = _extract_provider_error(http_result.parsed_json)
    if http_result.status_code in {401, 403}:
        return "auth_failed", error_message or f"Provider rejected the request with HTTP {http_result.status_code}."
    if http_result.status_code and http_result.status_code >= 400:
        return "provider_error", error_message or f"Provider returned HTTP {http_result.status_code}."
    return None


def _extract_provider_error(parsed_json: Any | None) -> str | None:
    if not isinstance(parsed_json, dict):
        return None
    error = parsed_json.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or error.get("status")
        if message:
            return str(message)
    if isinstance(error, str) and error:
        return error
    prompt_feedback = parsed_json.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        return f"Prompt blocked: {prompt_feedback['blockReason']}"
    if parsed_json.get("type") == "error" and parsed_json.get("message"):
        return str(parsed_json["message"])
    return None


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    pieces: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                pieces.append(text)
    return "\n".join(pieces).strip()


def _extract_gemini_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates", [])
    if candidates and isinstance(candidates[0], dict):
        metadata = candidates[0].get("urlContextMetadata")
        if isinstance(metadata, dict):
            return metadata
    metadata = payload.get("urlContextMetadata")
    return metadata if isinstance(metadata, dict) else {}


def _extract_claude_text_blocks(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for block in payload.get("content", []):
        if block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def _extract_claude_tool_use_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for block in payload.get("content", [])
        if isinstance(block, dict) and block.get("type") == "server_tool_use" and block.get("name") == "web_fetch"
    ]


def _extract_claude_fetch_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block in payload.get("content", []):
        if not isinstance(block, dict) or block.get("type") != "web_fetch_tool_result":
            continue
        content = block.get("content", {})
        if not isinstance(content, dict):
            continue
        document = content.get("content", {})
        results.append(
            {
                "url": content.get("url"),
                "title": document.get("title"),
                "document_type": document.get("type"),
                "media_type": ((document.get("source") or {}).get("media_type") if isinstance(document, dict) else None),
            }
        )
    return results


def _extract_openai_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    pieces: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                pieces.append(text)
    return "\n".join(pieces).strip()


def _extract_openai_citations(payload: dict[str, Any]) -> list[EvidenceRecord]:
    citations: list[EvidenceRecord] = []
    seen: set[tuple[str, str]] = set()
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            for annotation in content.get("annotations", []):
                citation_data = annotation.get("url_citation", {}) if isinstance(annotation, dict) else {}
                url = citation_data.get("url")
                title = citation_data.get("title") or url
                if isinstance(url, str) and url:
                    key = (url, str(title))
                    if key not in seen:
                        seen.add(key)
                        citations.append(EvidenceRecord(label=str(title), value=url, url=url))
    return citations


def _extract_openai_web_search_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in payload.get("output", []):
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action")
        actions.append(
            {
                "action": action,
                "id": item.get("id"),
                "status": item.get("status"),
                "queries": item.get("queries"),
                "action_type": action.get("type") if isinstance(action, dict) else None,
                "url": action.get("url") if isinstance(action, dict) else None,
            }
        )
    return actions


def _extract_openai_action_urls(actions: list[dict[str, Any]]) -> list[str]:
    return _unique_urls([action.get("url") for action in actions])


def _extract_structured_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        return {}
    for fenced in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL):
        parsed = _safe_json_loads(fenced)
        if isinstance(parsed, dict):
            return parsed
    parsed = _safe_json_loads(candidate)
    if isinstance(parsed, dict):
        return parsed
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    possible = candidate[start : end + 1]
    parsed = _safe_json_loads(possible)
    return parsed if isinstance(parsed, dict) else {}


def _classify_openai_diagnostic(
    payload: dict[str, Any],
    target_url: str,
    verification: VerificationResult,
    blocker_reason: str | None,
    extracted_title: str | None,
    extracted_heading: str | None,
    quotes: list[str],
    facts: list[str],
    actions: list[dict[str, Any]],
) -> tuple[str, str, float]:
    exact_verdict, exact_summary, exact_confidence = _classify_exact_access(
        provider_name=PRESETS["openai"].display_name,
        verification=verification,
        blocker_reason=blocker_reason,
        extracted_title=extracted_title,
        extracted_heading=extracted_heading,
        quotes=quotes,
        facts=facts,
    )

    if exact_verdict == "likely_blocked_by_site":
        return exact_verdict, exact_summary, exact_confidence

    incomplete_details = payload.get("incomplete_details")
    incomplete_reason = (
        incomplete_details.get("reason")
        if isinstance(incomplete_details, dict)
        else None
    )
    tool_open_urls = [
        action.get("url")
        for action in actions
        if action.get("action_type") in {"open_page", "find_in_page"}
    ]
    target_matched_by_tool_open = _normalize_url(target_url) in {
        _normalize_url(url) for url in tool_open_urls if _normalize_url(url)
    }

    if payload.get("status") == "incomplete" and incomplete_reason == "max_output_tokens":
        if target_matched_by_tool_open:
            return (
                "accessible",
                "OpenAI opened the exact target URL with web search. The response hit max_output_tokens before it returned a full extraction, but page access itself appears to have succeeded.",
                0.72,
            )
        return (
            "inconclusive",
            "OpenAI ran out of output tokens before it returned enough evidence to judge access confidently.",
            0.4,
        )

    if target_matched_by_tool_open:
        if exact_verdict == "accessible":
            return (
                "accessible",
                "OpenAI opened the exact target URL with web search and returned supporting page evidence.",
                0.88,
            )
        return (
            "accessible",
            "OpenAI opened the exact target URL with web search. Access appears successful even though extracted page details were limited.",
            0.76,
        )

    return exact_verdict, exact_summary, exact_confidence


def _verify_access(
    requested_url: str,
    candidate_urls: list[str],
    control: ControlFingerprintResult,
    extracted_title: str | None,
    extracted_heading: str | None,
    quotes: list[str],
    facts: list[str],
    blocker_reason: str | None,
) -> VerificationResult:
    accepted_urls = {
        value
        for value in (
            _normalize_url(requested_url),
            _normalize_url(control.final_url),
            _normalize_url(control.canonical_url),
        )
        if value
    }
    normalized_candidates = [_normalize_url(url) for url in candidate_urls if _normalize_url(url)]
    matched_url = next((url for url in normalized_candidates if url in accepted_urls), None)
    mismatch_notes: list[str] = []
    title_match: bool | None = None
    heading_match: bool | None = None
    quote_hits = 0

    if candidate_urls and matched_url is None:
        mismatch_notes.append("Provider evidence pointed to a different URL or did not confirm the requested URL.")

    if control.verdict == "available":
        if extracted_title and control.page_title:
            title_match = _text_matches(extracted_title, control.page_title)
            if title_match is False:
                mismatch_notes.append(
                    f"Title mismatch against control fingerprint: provider='{extracted_title}' vs control='{control.page_title}'."
                )
        if extracted_heading and control.main_heading:
            heading_match = _text_matches(extracted_heading, control.main_heading)
            if heading_match is False:
                mismatch_notes.append(
                    f"Heading mismatch against control fingerprint: provider='{extracted_heading}' vs control='{control.main_heading}'."
                )
        haystack = f"{control.raw_excerpt}\n{control.snippet}".lower()
        quote_hits = sum(1 for quote in quotes if quote and quote.lower() in haystack)
        if quotes and haystack and quote_hits == 0 and title_match is False and heading_match is False:
            mismatch_notes.append("Returned quotes were not found in the local control excerpt.")

    plausible_content = bool(extracted_title or extracted_heading or quotes or facts)
    block_signals = _collect_block_signals(
        extracted_title,
        extracted_heading,
        blocker_reason,
        *quotes,
        *facts,
    )

    confidence = 0.35
    if matched_url and plausible_content:
        confidence = 0.75
    if matched_url and (title_match or heading_match or quote_hits > 0):
        confidence = 0.92
    if not control.verdict == "available" and matched_url and plausible_content:
        confidence = 0.7
    if block_signals:
        confidence = max(confidence, 0.78)

    return VerificationResult(
        control_available=control.verdict == "available",
        matches_requested_url=matched_url is not None,
        matched_url=matched_url,
        title_match=title_match,
        heading_match=heading_match,
        quote_hits=quote_hits,
        plausible_content=plausible_content,
        confidence=round(confidence, 2),
        mismatch_notes=mismatch_notes,
        block_signals=block_signals,
    )


def _classify_exact_access(
    provider_name: str,
    verification: VerificationResult,
    blocker_reason: str | None,
    extracted_title: str | None,
    extracted_heading: str | None,
    quotes: list[str],
    facts: list[str],
) -> tuple[str, str, float]:
    has_content = bool(extracted_title or extracted_heading or quotes or facts)

    if blocker_reason and not verification.block_signals:
        block_confidence = 0.68 if not verification.matches_requested_url else 0.84
        return "likely_blocked_by_site", blocker_reason, block_confidence

    if verification.block_signals:
        signal_text = ", ".join(verification.block_signals[:3])
        summary = blocker_reason or f"{provider_name} appeared to hit a block, challenge, or interstitial page ({signal_text})."
        confidence = 0.9 if verification.matches_requested_url else 0.75
        return "likely_blocked_by_site", summary, confidence

    if verification.matches_requested_url and verification.plausible_content:
        if verification.control_available and (
            verification.title_match or verification.heading_match or verification.quote_hits > 0
        ):
            return "accessible", "Provider returned content that matches the requested URL and the local fingerprint.", 0.93
        if verification.control_available:
            return "accessible", "Provider returned content for the requested URL, though validation evidence was lighter.", 0.78
        return "accessible", "Provider returned content for the requested URL, but no local validation fingerprint was available.", 0.72

    if verification.mismatch_notes:
        return "inconclusive", "Provider returned some evidence, but it did not line up cleanly with the requested page.", 0.45

    if not has_content:
        return "inconclusive", "Provider did not return enough page evidence to confirm exact URL access.", 0.3

    return "inconclusive", "Provider returned partial evidence, but exact URL access could not be confirmed.", 0.4


def _build_evidence_records(extraction: dict[str, Any]) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    observed_url = _nullable_str(extraction.get("observed_url"))
    if observed_url:
        records.append(EvidenceRecord(label="Observed URL", value=observed_url, url=observed_url))
    page_title = _nullable_str(extraction.get("page_title"))
    if page_title:
        records.append(EvidenceRecord(label="Page title", value=page_title))
    main_heading = _nullable_str(extraction.get("main_heading"))
    if main_heading:
        records.append(EvidenceRecord(label="Main heading", value=main_heading))
    access_method = _nullable_str(extraction.get("access_method"))
    if access_method:
        records.append(EvidenceRecord(label="Access method", value=access_method))
    page_title_source = _nullable_str(extraction.get("page_title_source"))
    if page_title_source:
        records.append(EvidenceRecord(label="Page title source", value=page_title_source))
    blocker_reason = _nullable_str(extraction.get("blocker_reason"))
    if blocker_reason:
        records.append(EvidenceRecord(label="Blocker reason", value=blocker_reason))
    return records


def _extract_citations_from_object(obj: Any) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    seen: set[tuple[str, str]] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            possible_url = value.get("url") or value.get("uri") or value.get("href")
            possible_title = value.get("title") or possible_url
            if isinstance(possible_url, str) and possible_url.startswith(("http://", "https://")):
                key = (possible_url, str(possible_title))
                if key not in seen:
                    seen.add(key)
                    records.append(EvidenceRecord(label=str(possible_title), value=possible_url, url=possible_url))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            for match in URL_PATTERN.findall(value):
                key = (match, match)
                if key not in seen:
                    seen.add(key)
                    records.append(EvidenceRecord(label=match, value=match, url=match))

    walk(obj)
    return records


def _urls_to_evidence(urls: list[str]) -> list[EvidenceRecord]:
    return [EvidenceRecord(label=url, value=url, url=url) for url in _unique_urls(urls)]


def _parse_json(raw_body: str, content_type: str) -> Any | None:
    if not raw_body.strip():
        return None
    if "json" in content_type.lower() or raw_body.lstrip().startswith(("{", "[")):
        return _safe_json_loads(raw_body)
    return None


def _safe_json_loads(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _looks_like_html(content_type: str, raw_body: str) -> bool:
    lowered_type = content_type.lower()
    lowered_body = raw_body.lstrip().lower()
    return (
        "text/html" in lowered_type
        or lowered_body.startswith("<!doctype html")
        or lowered_body.startswith("<html")
        or lowered_body.startswith("<head")
    )


def _collect_block_signals(*values: str | None) -> list[str]:
    haystack = " ".join(value for value in values if value).lower()
    return [hint for hint in BLOCK_PAGE_HINTS if hint in haystack]


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                output.append(cleaned)
    return output[:2]


def _nullable_str(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.lower() in {"null", "none", "n/a", "na", "undefined"}:
            return None
        return cleaned
    return None


def _normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def _find_urls_in_object(obj: Any) -> list[str]:
    urls: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            urls.extend(URL_PATTERN.findall(value))

    walk(obj)
    return _unique_urls(urls)


def _first_url(urls: list[str]) -> str | None:
    for value in urls:
        normalized = _normalize_url(value)
        if normalized:
            return normalized
    return None


def _unique_urls(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = _normalize_url(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _text_matches(left: str, right: str) -> bool:
    normalized_left = _normalize_text(left).lower()
    normalized_right = _normalize_text(right).lower()
    if not normalized_left or not normalized_right:
        return False
    return normalized_left == normalized_right or normalized_left in normalized_right or normalized_right in normalized_left


def _normalize_text(value: str) -> str:
    return " ".join(unescape(value).split())


def _mask_secret(value: str) -> str:
    if not value:
        return value
    tail = value[-4:] if len(value) >= 4 else value
    return f"***{tail}"


def _format_ts(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


def _duration_ms(started_at: datetime, completed_at: datetime) -> int:
    return int((completed_at - started_at).total_seconds() * 1000)
