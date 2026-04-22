from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse


@dataclass(frozen=True)
class ProviderPreset:
    provider_id: str
    display_name: str
    default_api_url: str
    default_model: str
    docs_url: str
    description: str
    exact_url_supported: bool


@dataclass
class ProviderApiConfig:
    provider_id: str
    display_name: str
    api_url: str
    api_key: str
    model: str
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class APIRequestSpec:
    provider_id: str
    provider_name: str
    request_kind: str
    method: str
    url: str
    model: str
    headers: dict[str, str]
    json_body: dict[str, Any] | None
    timeout_seconds: float


PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        provider_id="openai",
        display_name="ChatGPT / OpenAI",
        default_api_url="https://api.openai.com/v1/responses",
        default_model="gpt-4.1-mini",
        docs_url="https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses",
        description="Search-based access check using OpenAI web search. Page-opening evidence matters more than perfect extraction.",
        exact_url_supported=False,
    ),
    "gemini": ProviderPreset(
        provider_id="gemini",
        display_name="Gemini / Google AI",
        default_api_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        default_model="gemini-2.5-flash",
        docs_url="https://ai.google.dev/gemini-api/docs/url-context",
        description="Primary exact-URL test using Gemini URL context.",
        exact_url_supported=True,
    ),
    "anthropic": ProviderPreset(
        provider_id="anthropic",
        display_name="Claude / Anthropic",
        default_api_url="https://api.anthropic.com/v1/messages",
        default_model="claude-sonnet-4-20250514",
        docs_url="https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool",
        description="Primary exact-URL test using Claude web fetch.",
        exact_url_supported=True,
    ),
}


def default_provider_configs() -> dict[str, ProviderApiConfig]:
    return {
        provider_id: ProviderApiConfig(
            provider_id=provider_id,
            display_name=preset.display_name,
            api_url=preset.default_api_url,
            api_key="",
            model=preset.default_model,
        )
        for provider_id, preset in PRESETS.items()
    }


def build_gemini_access_request(target_url: str, config: ProviderApiConfig) -> APIRequestSpec:
    return APIRequestSpec(
        provider_id=config.provider_id,
        provider_name=config.display_name,
        request_kind="exact_url_access",
        method="POST",
        url=_resolve_gemini_url(config.api_url, config.model),
        model=config.model,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": config.api_key.strip(),
        },
        json_body={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_exact_url_prompt(target_url)}],
                }
            ],
            "tools": [{"url_context": {}}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 900,
            },
        },
        timeout_seconds=config.timeout_seconds,
    )


def build_claude_access_request(target_url: str, config: ProviderApiConfig) -> APIRequestSpec:
    allowed_domain = _allowed_domain(target_url)
    return APIRequestSpec(
        provider_id=config.provider_id,
        provider_name=config.display_name,
        request_kind="exact_url_access",
        method="POST",
        url=config.api_url.strip(),
        model=config.model,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": config.api_key.strip(),
        },
        json_body={
            "model": config.model,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "user",
                    "content": build_exact_url_prompt(target_url),
                }
            ],
            "tools": [
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_uses": 1,
                    "allowed_domains": [allowed_domain] if allowed_domain else [],
                    "citations": {"enabled": True},
                }
            ],
        },
        timeout_seconds=config.timeout_seconds,
    )


def build_openai_diagnostic_request(target_url: str, config: ProviderApiConfig) -> APIRequestSpec:
    return APIRequestSpec(
        provider_id=config.provider_id,
        provider_name=config.display_name,
        request_kind="best_effort_diagnostic",
        method="POST",
        url=config.api_url.strip(),
        model=config.model,
        headers={
            "Authorization": f"Bearer {config.api_key.strip()}",
            "Content-Type": "application/json",
        },
        json_body={
            "model": config.model,
            "tools": [{"type": "web_search"}],
            "input": build_openai_diagnostic_prompt(target_url),
            "max_output_tokens": 1600,
        },
        timeout_seconds=config.timeout_seconds,
    )


def build_exact_url_prompt(target_url: str) -> str:
    return f"""
Use your URL-access capability to inspect the exact page at {target_url}.
Do not answer from memory or from general search results. Only report content you directly accessed from that URL.

Return exactly one JSON object with this shape:
{{
  "observed_url": "string or null",
  "page_title": "string or null",
  "main_heading": "exact text of the first H1 tag on the page, or null",
  "quotes": ["up to 2 short exact quotes from the page"],
  "facts": ["up to 2 concrete facts visible on the page"],
  "blocker_reason": "string or null"
}}

Rules:
- If you cannot access the exact page content, set blocker_reason and leave quotes empty.
- main_heading must be the exact text of the first H1 element only, not a summary or the page title.
- Keep quotes short, exact, and copied verbatim from the page if available.
- Do not include markdown fences.
""".strip()


def build_openai_diagnostic_prompt(target_url: str) -> str:
    return f"""
Use web search to determine whether your web-access tooling can find and inspect the exact URL {target_url}.
Your primary goal is to determine whether you could access or open that exact URL.
If you cannot confirm the exact URL itself, do not guess and do not treat the whole domain as equivalent.

Return exactly one JSON object with this shape:
{{
  "observed_url": "the exact URL you could inspect or cite, or null",
  "page_title": "the exact current page title string from the opened page, or null",
  "main_heading": "exact text of the first H1 tag on the page, or null",
  "quotes": ["up to 2 short exact quotes, if any"],
  "facts": ["up to 2 concrete facts, if any"],
  "blocker_reason": "string or null",
  "access_method": "opened_exact_url | cited_exact_url | related_results_only | null",
  "page_title_source": "opened_page_title | citation_metadata | null"
}}

Rules:
- Prefer null over guessing.
- If you opened the exact URL, copy the page title exactly as shown on the opened page. Do not shorten it, paraphrase it, modernize it, or answer from memory.
- If you cannot directly read the title from the opened page itself, set page_title to null.
- If you only found related search results but not the exact URL, set observed_url to null, page_title to null, and access_method to "related_results_only".
- Only use access_method="opened_exact_url" when you actually opened the target URL itself.
- main_heading must be the exact text of the first H1 element only, not a summary or the page title.
- Do not include markdown fences.
""".strip()


def _resolve_gemini_url(api_url: str, model: str) -> str:
    sanitized_model = quote(model.strip(), safe="-._~")
    template = api_url.strip()
    if "{model}" in template:
        return template.format(model=sanitized_model)
    return template


def _allowed_domain(target_url: str) -> str:
    parsed = urlparse(target_url)
    return parsed.netloc.lower()
