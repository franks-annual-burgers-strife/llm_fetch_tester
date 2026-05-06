# LLM URL Access Tester

Streamlit app for technical SEO checks: test whether major provider-managed LLM chat tools can access a supplied public URL.

The app is access-first. Page title, H1, quotes, facts, citations, and raw responses are supporting evidence, not the main verdict. This matters because providers can reach a URL while still reporting a normalized, cached, or heading-like representation of the page.

## Providers

- Gemini / Google AI: exact URL check using Gemini URL context.
- Claude / Anthropic: exact URL check using Claude web fetch.
- ChatGPT / OpenAI: search-based access check using the Responses API `web_search` tool.

Current defaults:

| Provider | Default model | Tool path | Notes |
| --- | --- | --- | --- |
| Gemini | `gemini-2.5-flash` | `url_context` | Google documents URL context for direct URL retrieval, with model support also including newer Gemini 3 models. |
| Claude | `claude-sonnet-4-20250514` | `web_fetch_20250910` | Anthropic's newer `web_fetch_20260209` adds dynamic filtering for newer Claude 4.6+ models; this app uses the still-available `20250910` tool path for broad compatibility. |
| OpenAI | `gpt-4.1-mini` | Responses API `web_search` | Search is required in the request and source-list inclusion is enabled. OpenAI documents newer recommended search paths such as `gpt-5.5`, but `gpt-4.1-mini` remains supported for Responses web search. |

Provider docs:

- [Gemini URL context](https://ai.google.dev/gemini-api/docs/url-context)
- [Claude web fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)
- [OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search)

## Local Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

For local test runs:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## How To Interpret Results

- `accessible`: the provider returned evidence tied to the requested URL.
- `likely_blocked_by_site`: the provider or extracted page evidence looked like a block, challenge, login wall, paywall, or interstitial.
- `inconclusive`: the provider returned too little or mismatched evidence to make a strong call.
- `auth_failed`: the provider API key was missing or rejected.
- `provider_error`: the provider API rejected the request or returned another API-level error.

The validation-only control fingerprint is a local HTTP fetch. It extracts lightweight reference data such as HTTP status, final URL, title, first literal HTML `h1`, canonical URL, and a snippet. It is used to compare provider evidence, but it is not the answer to whether a provider-managed chat tool can access the URL.

JavaScript fallback pages are treated as access issues when the provider reached the exact URL. For example, a page that returns "enable JavaScript to run this app" is different from a hard block, CAPTCHA, Cloudflare challenge, or provider tool error. It means the provider got through to the URL before rendering, but may only see limited server-rendered content.

Title, H1, quote, and fact mismatches should be read carefully. A provider may successfully access a page but return a prominent hero tagline, search metadata, cached title, or normalized document text instead of the literal DOM title or first `h1`. For this tool, exact URL access, provider retrieval metadata, citations, and block-page signals carry more weight than perfect content extraction.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository and use `app.py` as the entrypoint.
3. Keep `requirements.txt` in the repo root so Community Cloud can install dependencies.
4. In the app's Advanced settings, paste secrets based on `secrets.example.toml`.
5. If you prefer, you can still paste API keys directly into the sidebar at runtime instead of using secrets.

Example secrets:

```toml
[openai]
api_key = "your-openai-key"

[gemini]
api_key = "your-gemini-key"

[anthropic]
api_key = "your-anthropic-key"
```

## What It Shows

- whether each provider likely accessed the exact URL
- likely block or interstitial detection
- provider-reported evidence like title, heading, quotes, facts, and citations
- provider tool metadata, where exposed
- raw provider responses for debugging

## Notes

- API keys can be pasted into the UI for session-only use, or loaded from Streamlit secrets.
- `.streamlit/secrets.toml`, `.env`, `.venv`, caches, and Python bytecode are ignored by git.
- The local control fingerprint is validation-only, not the primary verdict.
- Claude web fetch currently does not support JavaScript-rendered-only websites; server-rendered HTML and PDFs are the strongest cases.
- Gemini URL context has provider-side limits and unsupported content types, including paywalled content.
- OpenAI's result is intentionally labelled search-based because OpenAI's documented API path is web search rather than a Gemini-style URL context tool or Claude-style web fetch tool.
