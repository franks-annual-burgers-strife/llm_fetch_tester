# LLM URL Access Tester

Streamlit app for checking whether major LLM chat tools can access a supplied URL.

## Providers

- Gemini: exact URL context check
- Claude: exact web fetch check
- OpenAI: search-based access check

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
- raw provider responses for debugging

## Notes

- API keys can be pasted into the UI for session-only use, or loaded from Streamlit secrets
- the local control fingerprint is validation-only, not the primary verdict
