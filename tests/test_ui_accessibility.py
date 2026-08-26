"""Accessibility properties that regressed once and must not regress again.

Deliberately not screenshot tests. Streamlit's generated DOM and class names
churn between releases, a visual baseline would need re-approving on every theme
tweak, and - decisively - a headless baseline would not have caught the defect
these tests exist for: the dark-mode failure lived in a browser-preference render
path that no fixed screenshot exercises.

What is asserted instead are the specific, checkable causes:

* the theme is pinned in supported configuration rather than left to the
  browser while CSS forces one half of the palette;
* no rule targets the app background, which is what made the two halves
  disagree;
* every colour pair the custom CSS introduces meets WCAG AA for its text size;
* headings descend without skipping a level;
* status and explanation text carry their meaning in words, not only in colour.
"""
import re

import pytest
import streamlit as st

from conftest import REPO_ROOT
from ui import public_components, theme

pytest.importorskip("streamlit.testing.v1", reason="streamlit testing API unavailable")
from streamlit.testing.v1 import AppTest

CONFIG_PATH = REPO_ROOT / ".streamlit" / "config.toml"

#: Sources allowed to contain styling. Anything else must not carry CSS.
STYLED_SOURCES = (REPO_ROOT / "ui" / "theme.py",)

#: Every source this project owns for the two front ends.
OWNED_SOURCES = (
    REPO_ROOT / "streamlit_app.py",
    REPO_ROOT / "admin_app.py",
    REPO_ROOT / "ui" / "theme.py",
    REPO_ROOT / "ui" / "formatting.py",
    REPO_ROOT / "ui" / "public_components.py",
    REPO_ROOT / "ui" / "admin_components.py",
)

WCAG_AA_NORMAL = 4.5
WCAG_AA_LARGE = 3.0


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.1 relative luminance for an #rrggbb colour."""
    digits = hex_colour.lstrip("#")
    channels = [int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def theme_config() -> dict[str, str]:
    """The [theme] table, parsed without assuming a TOML library version."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    section = text.split("[theme]", 1)[1]
    return {
        key.strip(): value.strip().strip('"')
        for key, _, value in (
            line.partition("=") for line in section.splitlines() if "=" in line
        )
        if not key.strip().startswith("#")
    }


# ================================================== the theme is pinned

def test_a_theme_configuration_file_exists():
    assert CONFIG_PATH.is_file(), "the theme must be pinned in supported configuration"


def test_the_theme_pins_an_explicit_base():
    """Unset base means Streamlit follows the browser's colour scheme."""
    assert theme_config()["base"] == "light"


@pytest.mark.parametrize(
    "setting",
    ["primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"],
)
def test_the_theme_defines_every_core_colour(setting):
    value = theme_config().get(setting, "")

    assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), f"{setting} is not an #rrggbb colour"


def test_no_source_overrides_the_application_background():
    """The dark-mode defect: forcing a background without a text colour."""
    for path in OWNED_SOURCES:
        source = path.read_text(encoding="utf-8")
        assert ".stApp" not in source, f"{path.name} targets the Streamlit app container"


def test_only_the_theme_module_carries_styling():
    for path in OWNED_SOURCES:
        if path in STYLED_SOURCES:
            continue
        assert "<style>" not in path.read_text(encoding="utf-8"), path.name


def test_no_source_targets_a_streamlit_generated_selector():
    """Those selectors are private and change between Streamlit releases."""
    for path in OWNED_SOURCES:
        source = path.read_text(encoding="utf-8")
        # The CSS attribute-selector form, so prose about never using one does
        # not trip the check.
        assert "[data-testid" not in source, path.name
        assert ".emotion-cache" not in source, path.name
        assert ".st-emotion" not in source, path.name


# ============================================================== contrast

def test_theme_text_is_readable_on_the_theme_background():
    config = theme_config()

    ratio = contrast_ratio(config["textColor"], config["backgroundColor"])
    assert ratio >= WCAG_AA_NORMAL, f"body text contrast is {ratio:.2f}:1"


def test_theme_text_is_readable_on_the_secondary_background():
    config = theme_config()

    ratio = contrast_ratio(config["textColor"], config["secondaryBackgroundColor"])
    assert ratio >= WCAG_AA_NORMAL, f"secondary surface contrast is {ratio:.2f}:1"


@pytest.mark.parametrize(
    "stop",
    [theme.HEADER_GRADIENT_START, theme.HEADER_GRADIENT_END],
    ids=["gradient_start", "gradient_end"],
)
def test_header_text_is_readable_across_the_whole_gradient(stop):
    """The old gradient ended on #00b4d8, where white measured 2.46:1."""
    ratio = contrast_ratio("#ffffff", stop)

    assert ratio >= WCAG_AA_NORMAL, f"white on {stop} is {ratio:.2f}:1"


@pytest.mark.parametrize(
    "fill",
    [theme.ELEVATED_FILL, theme.LOWER_FILL],
    ids=["elevated", "lower"],
)
def test_result_card_text_is_readable(fill):
    ratio = contrast_ratio(theme.TEXT_COLOR, fill)

    assert ratio >= WCAG_AA_NORMAL, f"card text on {fill} is {ratio:.2f}:1"


@pytest.mark.parametrize(
    ("edge", "fill"),
    [
        pytest.param(theme.ELEVATED_EDGE, theme.ELEVATED_FILL, id="elevated"),
        pytest.param(theme.LOWER_EDGE, theme.LOWER_FILL, id="lower"),
    ],
)
def test_result_card_edges_are_distinguishable_from_their_fill(edge, fill):
    """The coloured edge is decoration, so the 3:1 non-text minimum applies."""
    ratio = contrast_ratio(edge, fill)

    assert ratio >= WCAG_AA_LARGE, f"{edge} on {fill} is {ratio:.2f}:1"


# ====================================================== headings and status

def test_only_the_banner_emits_raw_heading_html():
    """Sections use st.header/st.subheader so levels cannot be skipped."""
    for path in OWNED_SOURCES:
        source = path.read_text(encoding="utf-8")
        headings = set(re.findall(r"<h([1-6])[ >]", source))
        if path.name == "theme.py":
            assert headings <= {"1"}, "the banner may emit an h1 and nothing else"
        else:
            assert not headings, f"{path.name} emits raw heading HTML"


def test_the_public_page_descends_h1_to_h2_to_h3_without_skipping():
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=180)
    app.run()

    # h1 from the banner, h2 from st.header, h3 from st.subheader.
    assert "<h1>" in "\n".join(str(block.value) for block in app.markdown)
    assert app.header, "expected at least one h2 section"
    assert app.subheader, "expected at least one h3 subsection"


def test_the_result_card_is_announced_as_a_live_region():
    assert 'role="status"' in theme.CUSTOM_CSS or 'role="status"' in (
        REPO_ROOT / "ui" / "theme.py"
    ).read_text(encoding="utf-8")


def test_the_scope_notice_precedes_any_interaction():
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=180)
    app.run()

    notices = " ".join(str(element.value) for element in app.info)
    assert "does not diagnose diabetes" in notices


# ================================================ meaning is not colour-only

def test_the_explanation_does_not_name_colours_to_carry_meaning():
    """The old legend read "Red = increases risk | Blue = decreases risk"."""
    source = (REPO_ROOT / "ui" / "public_components.py").read_text(encoding="utf-8")

    # Word boundaries matter: "entered =" contains the substring "red =".
    for phrase in (r"red\s*=", r"blue\s*=", r"green\s*=",
                   r"red\s+means", r"blue\s+means"):
        assert not re.search(phrase, source, re.IGNORECASE), f"colour-only legend: {phrase}"


def test_the_explanation_labels_direction_in_words():
    source = (REPO_ROOT / "ui" / "public_components.py").read_text(encoding="utf-8")

    assert "Increased the estimate" in source
    assert "Reduced the estimate" in source


def test_the_drift_verdict_is_written_as_text():
    source = (REPO_ROOT / "ui" / "admin_components.py").read_text(encoding="utf-8")

    assert '"YES"' in source
    assert '"No"' in source


def test_no_owned_source_encodes_status_in_an_emoji():
    """Emoji were the only marker for several verdicts before this track."""
    emoji = re.compile("[\U0001f300-\U0001faff☀-➿]")
    for path in OWNED_SOURCES:
        found = emoji.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path.name} still carries emoji markers: {found}"


def test_the_form_labels_every_input():
    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=180)
    st.cache_resource.clear()
    app.run()

    for widget in list(app.selectbox) + list(app.number_input):
        assert widget.label, "every input needs a visible label"
        assert widget.help, "every input needs help text"


def test_every_question_carries_help_text():
    for _heading, questions in public_components.SECTIONS:
        for question in questions:
            assert question.help.strip(), f"{question.feature} has no help text"
