from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Iterable

import streamlit as st

from probe_runner import (
    ControlFingerprintResult,
    ProviderAccessResult,
    TargetUrlRunConfig,
    TargetUrlRunResult,
    detect_code_language,
    normalize_target_url,
    run_url_access_suite,
    trim_text,
)
from providers import PRESETS, ProviderApiConfig, default_provider_configs
from version import __version__


st.set_page_config(
    page_title="LLM URL Access Tester",
    layout="wide",
)


def main() -> None:
    _inject_styles()
    _init_session_state()

    st.title(f"LLM URL Access Tester  `v{__version__}`")
    st.caption(
        "Test whether provider-managed chat tools can access a supplied URL. Gemini and Claude use exact direct-URL tools; OpenAI uses a search-based access check that looks for successful page opening."
    )
    st.info(
        "The local control fingerprint is validation-only. It helps compare returned titles, headings, and quotes, but it is not the primary answer to whether a provider could access the page."
    )

    target_col, action_col = st.columns([4, 1])
    with target_col:
        st.text_input(
            "Target URL",
            key="target_url",
            placeholder="https://example.com/article",
            help="Paste the exact public URL you want the chat tools to try to access.",
        )
    with action_col:
        run_clicked = st.button("Run All Chat Tool Checks", type="primary", use_container_width=True)

    _render_sidebar()

    if run_clicked:
        raw_target = st.session_state["target_url"].strip()
        if not raw_target:
            st.error("Enter a target URL before running the checks.")
        else:
            with st.spinner("Running provider-managed URL access checks..."):
                st.session_state["latest_result"] = run_url_access_suite(_build_run_config())

    result = _latest_result()

    top_col, report_col = st.columns([3, 1])
    with top_col:
        if result is not None:
            st.caption(f"Last run: {result.generated_at}")
    with report_col:
        st.download_button(
            "Download Markdown Report",
            data=_build_markdown_report(result),
            file_name=f"llm-url-access-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md",
            mime="text/markdown",
            disabled=result is None,
            use_container_width=True,
        )

    if result is None:
        st.warning("No URL access results yet. Enter a URL and run the checks.")
        return

    _render_summary_metrics(result)
    _render_control_fingerprint(result.control)

    for provider_id in ("gemini", "anthropic", "openai"):
        _render_provider_card(result.providers[provider_id])


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Advanced")
        st.slider(
            "Timeout (seconds)",
            min_value=5,
            max_value=60,
            key="timeout_seconds",
            help="Applies to the validation fingerprint and all provider checks.",
        )
        st.checkbox(
            "Run OpenAI search-based access check",
            key="run_openai_diagnostic",
            help="Runs an OpenAI web-search check that treats successful opening of the exact URL as the main access signal.",
        )
        st.checkbox(
            "Show request debug details",
            key="show_request_debug",
            help="Reveal raw request payloads and headers in provider cards.",
        )
        st.caption(
            "API keys can be pasted for session-only use, or loaded from Streamlit secrets. They are never written to the repository."
        )

        with st.expander("Provider Credentials & Models", expanded=True):
            for provider_id in ("gemini", "anthropic", "openai"):
                preset = PRESETS[provider_id]
                st.markdown(f"**{preset.display_name}**")
                st.text_input("API key", key=f"{provider_id}_api_key", type="password")
                st.text_input("Model", key=f"{provider_id}_model")
                st.text_input("API URL", key=f"{provider_id}_api_url")
                st.caption(preset.description)

        if st.button("Clear Saved Results", use_container_width=True):
            st.session_state["latest_result"] = None
            st.rerun()


def _render_summary_metrics(result: TargetUrlRunResult) -> None:
    st.markdown("### Run Summary")
    items = [
        (
            "Target URL",
            f"<a class='summary-value summary-link' href='{escape(result.target_url)}' target='_blank' rel='noopener noreferrer'>{escape(_compact_url(result.target_url))}</a>",
            "url",
        ),
        (
            "Control Fingerprint",
            escape(result.control.verdict.replace("_", " ").title()),
            _summary_tone(result.control.verdict),
        ),
        (
            "Gemini",
            escape(result.providers["gemini"].verdict.replace("_", " ").title()),
            _summary_tone(result.providers["gemini"].verdict),
        ),
        (
            "Claude",
            escape(result.providers["anthropic"].verdict.replace("_", " ").title()),
            _summary_tone(result.providers["anthropic"].verdict),
        ),
        (
            "OpenAI",
            escape(result.providers["openai"].verdict.replace("_", " ").title()),
            _summary_tone(result.providers["openai"].verdict),
        ),
    ]
    cards = []
    for label, value_html, tone in items:
        tone_class = f" summary-item-{tone}" if tone else ""
        cards.append(
            "<div class='summary-item"
            f"{tone_class}'>"
            f"<div class='summary-label'>{escape(label)}</div>"
            f"<div class='summary-value'>{value_html}</div>"
            "</div>"
        )
    st.markdown("<div class='summary-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_control_fingerprint(control: ControlFingerprintResult) -> None:
    with st.expander("Validation-Only Control Fingerprint", expanded=False):
        st.write(control.summary)
        st.json(
            {
                "requested_url": control.requested_url,
                "verdict": control.verdict,
                "status_code": control.status_code,
                "final_url": control.final_url,
                "content_type": control.content_type or "(none)",
                "page_title": control.page_title,
                "main_heading": control.main_heading,
                "canonical_url": control.canonical_url,
                "duration_ms": control.duration_ms,
                "error": control.error,
            }
        )
        if control.snippet:
            st.write("Snippet")
            st.code(control.snippet, language="text")
        if control.raw_excerpt:
            st.write("Raw excerpt")
            st.code(trim_text(control.raw_excerpt, limit=2200), language=detect_code_language(control.raw_excerpt))


def _render_provider_card(result: ProviderAccessResult) -> None:
    preset = PRESETS[result.provider_id]
    with st.container(border=True):
        head_col, meta_col = st.columns([2, 1])
        with head_col:
            st.subheader(result.provider_name)
            st.caption(f"{result.mode_label} | Model: {result.model}")
        with meta_col:
            st.markdown(f"[Provider docs]({preset.docs_url})")

        st.markdown(
            f"<div class='status-chip status-{result.verdict}'>{result.verdict.replace('_', ' ').title()}</div>",
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Confidence", f"{int(result.confidence * 100)}%")
        metric_col2.metric("Evidence URL", _compact_url(result.evidence_url or "n/a"))
        metric_col3.metric("HTTP status", str(result.response_status) if result.response_status is not None else "n/a")

        if result.secondary_summary:
            label = result.secondary_verdict.replace("_", " ").title() if result.secondary_verdict else "Skipped"
            st.info(f"Best-effort diagnostic: {label}. {result.secondary_summary}")

        summary_tab, evidence_tab, verification_tab, raw_tab = st.tabs(
            ["Summary", "Evidence", "Verification", "Raw Response"]
        )

        with summary_tab:
            st.markdown(
                f"<div class='summary-copy summary-{result.verdict}'>{escape(result.summary)}</div>",
                unsafe_allow_html=True,
            )
            _render_compact_details(
                [
                    ("Provider", result.provider_name, None),
                    ("Mode", result.mode_label, None),
                    ("Verdict", result.verdict.replace("_", " ").title(), result.verdict),
                    ("Confidence", f"{int(result.confidence * 100)}%", result.verdict),
                    ("Exact URL Tool", "Yes" if result.exact_url_supported else "Search-based", None),
                    ("Requested URL", result.requested_url, "url"),
                    ("Evidence URL", result.evidence_url or "n/a", "url" if result.evidence_url else None),
                    ("HTTP Status", str(result.response_status) if result.response_status is not None else "n/a", _http_status_tone(result.response_status)),
                    ("Blocker Reason", result.blocker_reason or "n/a", "likely_blocked_by_site" if result.blocker_reason else None),
                    ("Duration", f"{result.duration_ms} ms", None),
                    ("Error", result.error or "n/a", "provider_error" if result.error else None),
                ]
            )

        with evidence_tab:
            _render_compact_details(
                [
                    ("Page Title", result.extracted_title or "n/a", None),
                    ("Main Heading", result.extracted_heading or "n/a", None),
                    ("Quotes", _join_lines(result.quotes), None),
                    ("Facts", _join_lines(result.facts), None),
                    ("Blocker Reason", result.blocker_reason or "n/a", "likely_blocked_by_site" if result.blocker_reason else None),
                ]
            )
            if result.evidence_records:
                st.write("Provider-reported evidence")
                st.json([item.__dict__ for item in result.evidence_records])
            if result.citations:
                st.write("Citations / referenced URLs")
                st.json([item.__dict__ for item in result.citations])
            else:
                st.caption("No citations or provider-exposed URLs were captured.")
            if result.tool_metadata:
                st.write("Tool metadata")
                st.json(result.tool_metadata)

        with verification_tab:
            st.write("Comparison against the validation-only local fingerprint")
            st.json(result.verification.to_dict())
            if result.verification.mismatch_notes:
                st.write("Mismatch notes")
                for note in result.verification.mismatch_notes:
                    st.write(f"- {note}")
            elif result.verification.control_available:
                st.caption("No fingerprint mismatches were detected.")
            else:
                st.caption("No local fingerprint was available for strong comparison.")

        with raw_tab:
            if result.raw_response_json is not None:
                st.write("Raw provider response JSON")
                st.json(result.raw_response_json)
            elif result.raw_response_text:
                st.code(result.raw_response_text, language=detect_code_language(result.raw_response_text))
            else:
                st.info("No raw provider response was captured.")

            if result.raw_response_text and result.raw_response_json is not None:
                st.write("Model text extracted from the response")
                st.code(trim_text(result.raw_response_text, limit=2500), language=detect_code_language(result.raw_response_text))

            if st.session_state["show_request_debug"]:
                st.write("Request headers")
                st.json(result.request_headers)
                st.write("Request JSON")
                st.json(result.request_json or {})
                st.write("Response headers")
                st.json(result.response_headers)


def _build_run_config() -> TargetUrlRunConfig:
    timeout_seconds = float(st.session_state["timeout_seconds"])
    configs = {
        provider_id: ProviderApiConfig(
            provider_id=provider_id,
            display_name=PRESETS[provider_id].display_name,
            api_url=st.session_state[f"{provider_id}_api_url"].strip(),
            api_key=st.session_state[f"{provider_id}_api_key"],
            model=st.session_state[f"{provider_id}_model"].strip(),
            timeout_seconds=timeout_seconds,
        )
        for provider_id in PRESETS
    }
    return TargetUrlRunConfig(
        target_url=normalize_target_url(st.session_state["target_url"]),
        timeout_seconds=timeout_seconds,
        openai=configs["openai"],
        gemini=configs["gemini"],
        anthropic=configs["anthropic"],
        run_openai_diagnostic=bool(st.session_state["run_openai_diagnostic"]),
    )


def _build_markdown_report(result: TargetUrlRunResult | None) -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# LLM URL Access Tester Report",
        "",
        f"Generated: {timestamp}",
        "",
    ]
    if result is None:
        lines.append("No run results are available yet.")
        return "\n".join(lines)

    lines.extend(
        [
            f"Target URL: `{result.target_url}`",
            "",
            "## Validation-Only Control Fingerprint",
            "",
            f"- Verdict: `{result.control.verdict}`",
            f"- Summary: {result.control.summary}",
            f"- HTTP status: `{result.control.status_code}`",
            f"- Final URL: `{result.control.final_url}`",
            f"- Page title: `{result.control.page_title}`",
            f"- Main heading: `{result.control.main_heading}`",
            "",
        ]
    )

    for provider_id in ("gemini", "anthropic", "openai"):
        provider = result.providers[provider_id]
        lines.extend(_provider_to_markdown(provider))

    return "\n".join(lines)


def _provider_to_markdown(result: ProviderAccessResult) -> list[str]:
    lines = [
        f"## {result.provider_name}",
        "",
        f"- Mode: `{result.mode_label}`",
        f"- Primary verdict: `{result.verdict}`",
        f"- Confidence: `{result.confidence}`",
        f"- Summary: {result.summary}",
        f"- Evidence URL: `{result.evidence_url}`",
        f"- Extracted title: `{result.extracted_title}`",
        f"- Extracted heading: `{result.extracted_heading}`",
        f"- Blocker reason: `{result.blocker_reason}`",
        "",
    ]
    if result.secondary_summary:
        lines.extend(
            [
                f"- Secondary verdict: `{result.secondary_verdict}`",
                f"- Secondary summary: {result.secondary_summary}",
                "",
            ]
        )
    lines.extend(
        [
            "### Verification",
            "",
            "```json",
            _json_dump(result.verification.to_dict()),
            "```",
            "",
            "### Raw provider response",
            "",
            f"```{detect_code_language(result.raw_response_json if result.raw_response_json is not None else result.raw_response_text)}",
            trim_text(
                _json_dump(result.raw_response_json)
                if result.raw_response_json is not None
                else (result.raw_response_text or "(empty response)")
            ),
            "```",
            "",
        ]
    )
    return lines


def _json_dump(value) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)


def _compact_url(value: str) -> str:
    if len(value) <= 36:
        return value
    return f"{value[:33]}..."


def _latest_result() -> TargetUrlRunResult | None:
    return st.session_state["latest_result"]


def _summary_tone(value: str | None) -> str | None:
    if not value:
        return None
    if value == "available":
        return "accessible"
    if value == "unavailable":
        return "inconclusive"
    return value


def _render_compact_details(rows: list[tuple[str, str, str | None]]) -> None:
    html_rows: list[str] = []
    for label, value, tone in rows:
        tone_class = f" detail-value-{tone}" if tone else ""
        html_rows.append(
            "<div class='detail-row'>"
            f"<div class='detail-label'>{escape(label)}</div>"
            f"<div class='detail-value{tone_class}'>{escape(str(value))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='detail-card'>" + "".join(html_rows) + "</div>",
        unsafe_allow_html=True,
    )


def _http_status_tone(status_code: int | None) -> str | None:
    if status_code is None:
        return None
    if 200 <= status_code < 300:
        return "accessible"
    if status_code in {401, 403}:
        return "auth_failed"
    if status_code >= 400:
        return "provider_error"
    return "inconclusive"


def _join_lines(values: Iterable[str]) -> str:
    cleaned = [value.strip() for value in values if value.strip()]
    return "\n".join(cleaned) if cleaned else "n/a"


def _init_session_state() -> None:
    defaults = default_provider_configs()
    legacy_model_migrations = {
        "openai": {
            "gpt-5-mini": defaults["openai"].model,
            "gpt-5-mini-2025-08-07": defaults["openai"].model,
        },
        "gemini": {
            "gemini-2.5-flash-lite": defaults["gemini"].model,
        },
        "anthropic": {
            "claude-3-5-haiku-latest": defaults["anthropic"].model,
        },
    }
    st.session_state.setdefault("target_url", "")
    st.session_state.setdefault("timeout_seconds", 20)
    st.session_state.setdefault("run_openai_diagnostic", True)
    st.session_state.setdefault("show_request_debug", False)
    st.session_state.setdefault("latest_result", None)
    migrated_any_model = False
    for provider_id, config in defaults.items():
        st.session_state.setdefault(
            f"{provider_id}_api_key",
            _provider_secret(provider_id, "api_key") or config.api_key,
        )
        st.session_state.setdefault(
            f"{provider_id}_model",
            _provider_secret(provider_id, "model") or config.model,
        )
        st.session_state.setdefault(
            f"{provider_id}_api_url",
            _provider_secret(provider_id, "api_url") or config.api_url,
        )
        current_model = st.session_state.get(f"{provider_id}_model")
        migrated_model = legacy_model_migrations.get(provider_id, {}).get(current_model)
        if migrated_model:
            st.session_state[f"{provider_id}_model"] = migrated_model
            migrated_any_model = True
    if migrated_any_model:
        st.session_state["latest_result"] = None


def _provider_secret(provider_id: str, field: str) -> str | None:
    try:
        provider_section = st.secrets.get(provider_id)
        if isinstance(provider_section, dict):
            value = provider_section.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        flat_value = st.secrets.get(f"{provider_id}_{field}")
        if isinstance(flat_value, str) and flat_value.strip():
            return flat_value.strip()
    except Exception:
        return None
    return None


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(241, 175, 95, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(100, 186, 180, 0.12), transparent 24%),
                #f7f4ee;
        }
        .status-chip {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid rgba(0, 0, 0, 0.08);
            background: #ffffff;
        }
        .status-accessible {
            color: #0f5132;
            background: #d1f1dd;
        }
        .status-likely_blocked_by_site {
            color: #842029;
            background: #f8d7da;
        }
        .status-inconclusive,
        .status-unsupported {
            color: #664d03;
            background: #fff3cd;
        }
        .status-auth_failed,
        .status-provider_error {
            color: #5d1f1f;
            background: #f0d3d3;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.4rem 0 1rem;
        }
        .summary-item {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 1rem;
            padding: 0.8rem 0.9rem;
            min-width: 0;
        }
        .summary-label {
            font-size: 0.75rem;
            line-height: 1.2;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #6c665c;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .summary-grid .summary-value {
            font-size: 0.92rem;
            line-height: 1.2;
            font-weight: 700;
            color: #26263a;
            word-break: break-word;
        }
        .summary-grid .summary-link {
            color: #0d5f73;
            text-decoration: none;
        }
        .summary-grid .summary-link:hover {
            text-decoration: underline;
        }
        .summary-item-accessible .summary-value {
            color: #0f5132;
        }
        .summary-item-likely_blocked_by_site .summary-value {
            color: #842029;
        }
        .summary-item-inconclusive .summary-value,
        .summary-item-unsupported .summary-value {
            color: #8a6500;
        }
        .summary-item-auth_failed .summary-value,
        .summary-item-provider_error .summary-value {
            color: #842029;
        }
        @media (max-width: 1100px) {
            .summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
        .summary-copy {
            font-size: 0.92rem;
            line-height: 1.45;
            padding: 0.8rem 0.95rem;
            border-radius: 0.85rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(0, 0, 0, 0.08);
            background: rgba(255, 255, 255, 0.7);
        }
        .summary-accessible {
            background: rgba(209, 241, 221, 0.75);
        }
        .summary-likely_blocked_by_site {
            background: rgba(248, 215, 218, 0.8);
        }
        .summary-inconclusive,
        .summary-unsupported {
            background: rgba(255, 243, 205, 0.8);
        }
        .summary-auth_failed,
        .summary-provider_error {
            background: rgba(240, 211, 211, 0.8);
        }
        .detail-card {
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 0.9rem;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.72);
        }
        .detail-row {
            display: grid;
            grid-template-columns: 9rem 1fr;
            gap: 0.8rem;
            padding: 0.62rem 0.85rem;
            border-top: 1px solid rgba(0, 0, 0, 0.05);
        }
        .detail-row:first-child {
            border-top: none;
        }
        .detail-label {
            font-size: 0.73rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #6c665c;
            font-weight: 700;
        }
        .detail-value {
            font-size: 0.84rem;
            line-height: 1.35;
            white-space: pre-wrap;
            word-break: break-word;
            color: #1f1f1b;
        }
        .detail-value-accessible {
            color: #0f5132;
            font-weight: 700;
        }
        .detail-value-likely_blocked_by_site {
            color: #842029;
            font-weight: 700;
        }
        .detail-value-inconclusive,
        .detail-value-unsupported {
            color: #8a6500;
            font-weight: 700;
        }
        .detail-value-auth_failed,
        .detail-value-provider_error {
            color: #842029;
            font-weight: 700;
        }
        .detail-value-url {
            color: #0d5f73;
        }
        .stCode pre, .stCode code, .stJson pre {
            font-size: 0.78rem !important;
            line-height: 1.35 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
