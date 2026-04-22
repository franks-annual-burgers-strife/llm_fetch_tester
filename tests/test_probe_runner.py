import json

import httpx

from probe_runner import (
    ControlFingerprintResult,
    fetch_control_fingerprint,
    run_claude_access_check,
    run_gemini_access_check,
    run_openai_access_check,
)
from providers import ProviderApiConfig


def _control_available() -> ControlFingerprintResult:
    return ControlFingerprintResult(
        requested_url="https://example.com/page",
        verdict="available",
        summary="Validation-only local fingerprint fetched with HTTP 200.",
        started_at="2026-04-21T15:00:00+10:00",
        completed_at="2026-04-21T15:00:00+10:00",
        duration_ms=20,
        status_code=200,
        final_url="https://example.com/page",
        content_type="text/html",
        page_title="Example Domain",
        main_heading="Example Domain",
        canonical_url="https://example.com/page",
        snippet="Example Domain. This domain is for use in illustrative examples in documents.",
        raw_excerpt="""
            <html><head><title>Example Domain</title></head>
            <body><h1>Example Domain</h1>
            <p>This domain is for use in illustrative examples in documents.</p>
            </body></html>
        """,
        response_headers={},
        error=None,
    )


def _control_unavailable() -> ControlFingerprintResult:
    return ControlFingerprintResult(
        requested_url="https://example.com/page",
        verdict="unavailable",
        summary="Validation-only local fingerprint could not be fetched.",
        started_at="2026-04-21T15:00:00+10:00",
        completed_at="2026-04-21T15:00:00+10:00",
        duration_ms=20,
        status_code=None,
        final_url=None,
        content_type="",
        page_title=None,
        main_heading=None,
        canonical_url=None,
        snippet="",
        raw_excerpt="",
        response_headers={},
        error="timeout",
    )


def test_fetch_control_fingerprint_extracts_title_heading_and_canonical() -> None:
    html = """
    <html>
      <head>
        <title>Example Domain</title>
        <link rel="canonical" href="https://example.com/page" />
      </head>
      <body>
        <h1>Example Domain</h1>
        <p>This domain is for use in illustrative examples in documents.</p>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = fetch_control_fingerprint("https://example.com/page", client=client)

    assert result.verdict == "available"
    assert result.page_title == "Example Domain"
    assert result.main_heading == "Example Domain"
    assert result.canonical_url == "https://example.com/page"


def test_gemini_accessible_when_metadata_and_content_match() -> None:
    config = ProviderApiConfig(
        provider_id="gemini",
        display_name="Gemini / Google AI",
        api_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        api_key="gemini-secret",
        model="gemini-2.5-flash",
    )
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "observed_url": "https://example.com/page",
                                    "page_title": "Example Domain",
                                    "main_heading": "Example Domain",
                                    "quotes": ["This domain is for use in illustrative examples in documents."],
                                    "facts": ["The page is an example domain."],
                                    "blocker_reason": None,
                                }
                            )
                        }
                    ]
                },
                "urlContextMetadata": {
                    "retrievedUrl": "https://example.com/page",
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_gemini_access_check(config, "https://example.com/page", _control_available(), client=client)

    assert result.verdict == "accessible"
    assert result.evidence_url == "https://example.com/page"
    assert result.verification.matches_requested_url is True
    assert result.verification.title_match is True


def test_gemini_still_accessible_without_control_fingerprint() -> None:
    config = ProviderApiConfig(
        provider_id="gemini",
        display_name="Gemini / Google AI",
        api_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        api_key="gemini-secret",
        model="gemini-2.5-flash",
    )
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "observed_url": "https://example.com/page",
                                    "page_title": "Example Domain",
                                    "main_heading": "Example Domain",
                                    "quotes": [],
                                    "facts": ["The page identifies itself as Example Domain."],
                                    "blocker_reason": None,
                                }
                            )
                        }
                    ]
                },
                "urlContextMetadata": {
                    "retrievedUrl": "https://example.com/page",
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_gemini_access_check(config, "https://example.com/page", _control_unavailable(), client=client)

    assert result.verdict == "accessible"
    assert result.confidence < 0.8


def test_claude_detects_block_page() -> None:
    config = ProviderApiConfig(
        provider_id="anthropic",
        display_name="Claude / Anthropic",
        api_url="https://api.anthropic.com/v1/messages",
        api_key="anthropic-secret",
        model="claude-sonnet-4-20250514",
    )
    payload = {
        "content": [
            {
                "type": "server_tool_use",
                "name": "web_fetch",
                "id": "tool_1",
            },
            {
                "type": "web_fetch_tool_result",
                "content": {
                    "url": "https://blocked.example.com/page",
                    "content": {
                        "type": "document",
                        "title": "Attention Required! | Cloudflare",
                        "source": {"media_type": "text/html"},
                    },
                },
            },
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "observed_url": "https://blocked.example.com/page",
                        "page_title": "Attention Required! | Cloudflare",
                        "main_heading": "Please verify you are human",
                        "quotes": [],
                        "facts": [],
                        "blocker_reason": "Cloudflare challenge page",
                    }
                ),
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_claude_access_check(config, "https://blocked.example.com/page", _control_unavailable(), client=client)

    assert result.verdict == "likely_blocked_by_site"
    assert "Cloudflare" in (result.blocker_reason or "")


def test_openai_accessible_when_exact_url_is_cited_and_content_matches() -> None:
    config = ProviderApiConfig(
        provider_id="openai",
        display_name="ChatGPT / OpenAI",
        api_url="https://api.openai.com/v1/responses",
        api_key="openai-secret",
        model="gpt-4.1-mini",
    )
    extraction = json.dumps(
        {
            "observed_url": "https://example.com/page",
            "page_title": "Example Domain",
            "main_heading": "Example Domain",
            "quotes": ["This domain is for use in illustrative examples in documents."],
            "facts": ["The page is an example domain."],
            "blocker_reason": None,
        }
    )
    payload = {
        "output_text": extraction,
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": extraction,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://example.com/page",
                                    "title": "Example Domain",
                                },
                            }
                        ],
                    }
                ],
            },
            {
                "type": "web_search_call",
                "action": "search",
                "status": "completed",
                "queries": ["example.com page"],
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_openai_access_check(config, "https://example.com/page", _control_available(), client=client)

    assert result.verdict == "accessible"
    assert result.secondary_verdict is None
    assert result.citations[0].url == "https://example.com/page"


def test_gemini_auth_failure() -> None:
    config = ProviderApiConfig(
        provider_id="gemini",
        display_name="Gemini / Google AI",
        api_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        api_key="gemini-secret",
        model="gemini-2.5-flash",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_gemini_access_check(config, "https://example.com/page", _control_available(), client=client)

    assert result.verdict == "auth_failed"
    assert "Invalid API key" in result.summary


def test_openai_incomplete_open_page_still_counts_as_accessible() -> None:
    config = ProviderApiConfig(
        provider_id="openai",
        display_name="ChatGPT / OpenAI",
        api_url="https://api.openai.com/v1/responses",
        api_key="openai-secret",
        model="gpt-4.1-mini",
    )
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "open_page",
                    "url": "https://example.com/page",
                },
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_openai_access_check(config, "https://example.com/page", _control_available(), client=client)

    assert result.verdict == "accessible"
    assert result.secondary_verdict is None
    assert "hit max_output_tokens" in result.summary
    assert result.evidence_url == "https://example.com/page"


def test_openai_provider_error_does_not_claim_api_endpoint_as_evidence_url() -> None:
    config = ProviderApiConfig(
        provider_id="openai",
        display_name="ChatGPT / OpenAI",
        api_url="https://api.openai.com/v1/responses",
        api_key="openai-secret",
        model="gpt-4.1-mini",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid tool configuration"}},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_openai_access_check(config, "https://example.com/page", _control_available(), client=client)

    assert result.verdict == "provider_error"
    assert result.secondary_verdict is None
    assert result.evidence_url is None
    assert "Invalid tool configuration" in result.summary


def test_openai_string_null_blocker_reason_is_not_treated_as_blocked() -> None:
    config = ProviderApiConfig(
        provider_id="openai",
        display_name="ChatGPT / OpenAI",
        api_url="https://api.openai.com/v1/responses",
        api_key="openai-secret",
        model="gpt-4.1-mini",
    )
    extraction = json.dumps(
        {
            "observed_url": "https://example.com/page",
            "page_title": "Example Domain",
            "main_heading": None,
            "quotes": [],
            "facts": ["Example Domain is reachable."],
            "blocker_reason": "null",
        }
    )
    payload = {
        "output_text": extraction,
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": extraction,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://example.com/page",
                                    "title": "Example Domain",
                                },
                            }
                        ],
                    }
                ],
            },
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "open_page",
                    "url": "https://example.com/page",
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = run_openai_access_check(config, "https://example.com/page", _control_available(), client=client)

    assert result.verdict == "accessible"
    assert result.blocker_reason is None
    assert result.summary != "null"
