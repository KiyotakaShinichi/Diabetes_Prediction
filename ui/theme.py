"""The small remainder of custom styling, and the cards that depend on it.

Global theme settings are NOT here. Base theme, background, text and primary
colours live in ``.streamlit/config.toml``, which is supported configuration and
applies to both entrypoints. This module holds only what Streamlit has no
primitive for: the page banner and the result hero card.

Two rules keep it from growing back into a styling framework:

* no Streamlit-generated class or ``data-testid`` is ever targeted, so a
  Streamlit upgrade cannot silently restyle the page;
* the app background is never overridden - doing that without also setting a
  text colour is what previously made the app unreadable for visitors whose
  browser asked for a dark theme.

Colours below are stated explicitly rather than inherited, and every foreground
/background pair is at or above WCAG AA for its text size.
"""
from __future__ import annotations

import streamlit as st

#: Banner gradient. White text measures 6.92:1 on #015f8f and 6.31:1 on #00697f,
#: both comfortably above the 4.5:1 normal-text minimum. The previous gradient
#: ended on #00b4d8, where white measured 2.46:1.
HEADER_GRADIENT_START = "#015f8f"
HEADER_GRADIENT_END = "#00697f"

#: Result card fills. Body text is #1f2933 on all of them: 12.08:1 and 13.43:1.
ELEVATED_FILL = "#fee2e2"
ELEVATED_EDGE = "#b91c1c"
LOWER_FILL = "#dcfce7"
LOWER_EDGE = "#15803d"

TEXT_COLOR = "#1f2933"

CUSTOM_CSS = f"""
<style>
    .app-header {{
        background: linear-gradient(135deg, {HEADER_GRADIENT_START}, {HEADER_GRADIENT_END});
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        color: #ffffff;
        margin-bottom: 1.25rem;
    }}

    .app-header h1 {{
        margin: 0;
        font-size: 1.6rem;
        color: #ffffff;
    }}

    .app-header p {{
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #ffffff;
    }}

    .risk-hero {{
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 6px solid;
        color: {TEXT_COLOR};
    }}

    .risk-hero .risk-label {{
        font-size: 0.95rem;
        margin: 0;
    }}

    .risk-hero .risk-value {{
        font-size: 2.75rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.2rem 0;
    }}

    .risk-hero .risk-band {{
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }}

    .risk-hero .risk-note {{
        font-size: 0.9rem;
        margin: 0.6rem 0 0 0;
    }}

    .risk-hero.is-elevated {{
        background-color: {ELEVATED_FILL};
        border-left-color: {ELEVATED_EDGE};
    }}

    .risk-hero.is-lower {{
        background-color: {LOWER_FILL};
        border-left-color: {LOWER_EDGE};
    }}
</style>
"""


def inject_css() -> None:
    """Apply the custom styles. Called once, from the entrypoint's main()."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def banner(title: str, subtitle: str) -> None:
    """The page banner, and the only <h1> either app emits.

    Raw heading HTML is used here and nowhere else: Streamlit has no banner
    primitive, and st.title would not carry the gradient. Every section below a
    banner uses st.header/st.subheader, so the document still runs h1 -> h2 -> h3
    without a skipped level.
    """
    st.markdown(
        f'<div class="app-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def risk_hero(*, label: str, value: str, band: str, note: str, elevated: bool) -> None:
    """The single primary result card.

    Marked as a live region so a screen reader announces the estimate when it
    appears after submission, rather than leaving it to be discovered.
    """
    modifier = "is-elevated" if elevated else "is-lower"
    st.markdown(
        f'<div class="risk-hero {modifier}" role="status" aria-live="polite">'
        f'<p class="risk-label">{label}</p>'
        f'<p class="risk-value">{value}</p>'
        f'<p class="risk-band">{band}</p>'
        f'<p class="risk-note">{note}</p>'
        "</div>",
        unsafe_allow_html=True,
    )
