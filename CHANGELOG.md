# Changelog

## [0.3.2] - 2026-05-15

### Changed
- Control fingerprint fetch now uses a browser-like User-Agent, Accept, and Accept-Language so generic-browser-accessible sites no longer return WAF challenge pages as the control.

## [0.3.1] - 2026-05-06

### Changed
- Local control fingerprint WAF/firewall responses now show as blocked validation warnings instead of available fingerprints.

## [0.3.0] - 2026-05-06

### Changed
- Updated project documentation to reflect current provider defaults, Streamlit Community Cloud usage, and provider-specific interpretation caveats.
- OpenAI web search requests now require tool use and request source-list inclusion to make the access check less dependent on model choice.
- JavaScript fallback pages on the exact requested URL are now classified as accessible with an access issue, rather than likely blocked.

### Fixed
- OpenAI citation parsing now handles both nested `url_citation` annotations and top-level URL citation annotations.
- OpenAI verification now considers URLs exposed in web search source metadata, not only citations and explicit `open_page` actions.
- Claude web fetch tool-level errors inside HTTP 200 responses now surface as likely blocked results instead of weak inconclusive results.

## [0.2.0] - 2026-04-22

### Changed
- Prompts now explicitly ask LLMs for the exact text of the first H1 tag rather than a generic "main heading", improving extraction precision.

## [0.1.0] - 2026-04-22

### Added
- Initial release: Streamlit app testing Gemini, Claude, and OpenAI URL access via provider-managed chat tools.
- Control fingerprint fetch for local validation of titles, headings, quotes, and facts.
- Sidebar API key/model/URL configuration with Streamlit secrets support.
