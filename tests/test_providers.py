from providers import (
    ProviderApiConfig,
    build_claude_access_request,
    build_gemini_access_request,
    build_openai_diagnostic_request,
)


def test_build_gemini_access_request() -> None:
    config = ProviderApiConfig(
        provider_id="gemini",
        display_name="Gemini / Google AI",
        api_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        api_key="gemini-secret",
        model="gemini-2.5-flash",
    )

    request = build_gemini_access_request("https://example.com/page", config)

    assert request.method == "POST"
    assert request.url.endswith("/models/gemini-2.5-flash:generateContent")
    assert request.headers["x-goog-api-key"] == "gemini-secret"
    assert request.json_body["tools"] == [{"url_context": {}}]
    assert request.json_body["generationConfig"]["maxOutputTokens"] == 900
    assert "responseMimeType" not in request.json_body["generationConfig"]
    assert "https://example.com/page" in request.json_body["contents"][0]["parts"][0]["text"]


def test_build_claude_access_request() -> None:
    config = ProviderApiConfig(
        provider_id="anthropic",
        display_name="Claude / Anthropic",
        api_url="https://api.anthropic.com/v1/messages",
        api_key="anthropic-secret",
        model="claude-sonnet-4-20250514",
    )

    request = build_claude_access_request("https://example.com/articles/test", config)

    assert request.method == "POST"
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "anthropic-secret"
    tool = request.json_body["tools"][0]
    assert tool["type"] == "web_fetch_20250910"
    assert tool["allowed_domains"] == ["example.com"]
    assert tool["citations"] == {"enabled": True}


def test_build_openai_diagnostic_request() -> None:
    config = ProviderApiConfig(
        provider_id="openai",
        display_name="ChatGPT / OpenAI",
        api_url="https://api.openai.com/v1/responses",
        api_key="openai-secret",
        model="gpt-4.1-mini",
    )

    request = build_openai_diagnostic_request("https://example.com/page", config)

    assert request.method == "POST"
    assert request.url == "https://api.openai.com/v1/responses"
    assert request.headers["Authorization"] == "Bearer openai-secret"
    assert request.json_body["tools"] == [{"type": "web_search"}]
    assert request.json_body["model"] == "gpt-4.1-mini"
    assert request.json_body["max_output_tokens"] == 1600
    assert "reasoning" not in request.json_body
    assert "text" not in request.json_body
    assert "https://example.com/page" in request.json_body["input"]
    assert "copy the page title exactly as shown on the opened page" in request.json_body["input"]
    assert "\"page_title_source\"" in request.json_body["input"]
