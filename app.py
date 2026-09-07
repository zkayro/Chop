import streamlit as st
import subprocess
import sys
import json
import re
import os
import hashlib
import math
import uuid
import html
import time
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from boundary_engine import (
    apply_to_candidates as apply_claim_aware_boundaries,
    find_natural_ending as engine_find_natural_ending,
    resolve_boundaries,
    align_start_to_words,
    align_end_to_words,
)

from ranking_calibration import (
    apply_quality_guardrails as calibrated_quality_guardrails,
    apply_score_confidence_caps as calibrated_score_confidence_caps,
    calibrated_scoring_bands_prompt,
    decide_publishability,
    finalize_relative_ranks,
    global_best_moments_prompt_block,
    quality_tier_for_score as calibrated_quality_tier_for_score,
)

st.set_page_config(
    page_title="mariundjenson",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "projects"
ANALYSIS_PIPELINE_VERSION = "two-stage-v2"
CANDIDATE_PIPELINE_VERSION = "region-variants-v2-semantic-v1"
AI_RANKING_VERSION = "viral-short-v8-global-best-moments"
AI_SCORING_CALIBRATION = AI_RANKING_VERSION
AI_CARD_OUTPUT_VERSION = "content-card-v3-auto-hook"
EXPERIMENTAL_REVIEW_SCORE_FLOOR = 30
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}

if "current_view" not in st.session_state:
    requested_view = st.query_params.get("view", "home")
    st.session_state.current_view = (
        requested_view if requested_view in {"home", "projects", "history"} else "home"
    )

if "project_dir" not in st.session_state:
    requested_project = st.query_params.get("project")
    safe_project_name = (
        requested_project
        and requested_project not in {".", ".."}
        and Path(requested_project).name == requested_project
    )
    candidate = PROJECTS_DIR / requested_project if safe_project_name else None
    st.session_state.project_dir = (
        str(candidate.resolve())
        if candidate and candidate.is_dir()
        and candidate.resolve().parent == PROJECTS_DIR.resolve()
        else None
    )


def create_project_dir():
    project_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=False)
    return project_dir


def get_project_dir():
    value = st.session_state.project_dir
    if not value:
        return None
    project_dir = Path(value).resolve()
    if project_dir.is_dir() and project_dir.parent == PROJECTS_DIR.resolve():
        return project_dir
    st.session_state.project_dir = None
    return None


def set_active_project(project_dir):
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir() or project_dir.parent != PROJECTS_DIR.resolve():
        raise ValueError("Invalid project directory")
    st.session_state.project_dir = str(project_dir)
    st.query_params["project"] = project_dir.name


def set_view(view):
    if view not in {"home", "projects", "history"}:
        raise ValueError("Invalid view")
    st.session_state.current_view = view
    st.query_params["view"] = view


def split_campaign_values(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def format_compact_number(value):
    number = max(0, int(value or 0))
    if number >= 1_000_000:
        compact = f"{number / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}M"
    if number >= 1_000:
        compact = f"{number / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}K"
    return str(number)


def save_campaign(project_dir, campaign):
    target = project_dir / "campaign.json"
    temporary = project_dir / "campaign.json.tmp"
    temporary.write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)


def normalize_ranking_options(options=None):
    options = options if isinstance(options, dict) else {}
    return {
        "viral_shorts": bool(options.get("viral_shorts", True)),
        "two_second_hook": bool(options.get("two_second_hook", True)),
        "candidate_debug_frames": bool(options.get("candidate_debug_frames", True)),
    }


def save_ranking_options(project_dir, options):
    target = project_dir / "ranking_options.json"
    temporary = project_dir / "ranking_options.json.tmp"
    temporary.write_text(
        json.dumps(normalize_ranking_options(options), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)


def save_import_metadata(project_dir, title, source_type, source_url=None):
    metadata = {
        "title": title,
        "source_type": source_type,
        "source_url": source_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = project_dir / "metadata.json"
    temporary = project_dir / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)


def save_uploaded_video(uploaded_file, project_dir, source_type="local_file"):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video format. Use MP4, MOV, MKV, or WEBM.")

    target = project_dir / f"source{suffix}"
    temporary = project_dir / f"source{suffix}.part"
    uploaded_file.seek(0)
    try:
        with temporary.open("wb") as destination:
            shutil.copyfileobj(uploaded_file, destination, length=1024 * 1024)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    if temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("The uploaded video file is empty.")
    temporary.replace(target)
    save_import_metadata(project_dir, uploaded_file.name, source_type)
    return target.resolve()


def is_direct_google_drive_url(url):
    try:
        parsed = urllib.parse.urlparse((url or "").strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https":
        return False
    if host == "drive.usercontent.google.com":
        return True
    if host not in {"drive.google.com", "www.drive.google.com"}:
        return False
    query = urllib.parse.parse_qs(parsed.query)
    return parsed.path.rstrip("/") == "/uc" and query.get("export") == ["download"]


def _drive_filename(headers, url):
    disposition = headers.get("Content-Disposition", "")
    encoded = re.search(r"filename\*=UTF-8''([^;]+)", disposition, flags=re.I)
    quoted = re.search(r'filename="?([^";]+)', disposition, flags=re.I)
    if encoded:
        return urllib.parse.unquote(encoded.group(1))
    if quoted:
        return quoted.group(1).strip()
    return Path(urllib.parse.urlparse(url).path).name


def download_direct_drive_video(url, project_dir):
    if not is_direct_google_drive_url(url):
        raise ValueError(
            "This Google Drive link is not a direct download link. "
            "Download the campaign file from Google Drive and upload it here."
        )

    request = urllib.request.Request(
        url.strip(),
        headers={"User-Agent": "Chopify/1.0"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise ValueError(
            "Google Drive could not download this file. Check that the direct link is publicly "
            "accessible, or download the campaign file and upload it here."
        ) from error

    with response:
        content_type = response.headers.get_content_type().lower()
        if content_type in {"text/html", "application/xhtml+xml"}:
            raise ValueError(
                "Google Drive returned a sign-in or sharing page instead of a video. "
                "Download the campaign file from Google Drive and upload it here."
            )
        filename = _drive_filename(response.headers, response.geturl())
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            suffix = VIDEO_CONTENT_TYPES.get(content_type)
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                "The direct Google Drive download is not a supported MP4, MOV, MKV, or WEBM file."
            )

        target = project_dir / f"source{suffix}"
        temporary = project_dir / f"source{suffix}.part"
        try:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        if temporary.stat().st_size <= 0:
            temporary.unlink(missing_ok=True)
            raise ValueError("Google Drive returned an empty file.")
        temporary.replace(target)

    title = filename or f"Google Drive campaign{suffix}"
    save_import_metadata(project_dir, title, "google_drive", url.strip())
    return target.resolve()

# --------------------------------------------------
# STYLE
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --chopify-bg: #070b12;
        --chopify-panel: #101722;
        --chopify-card: #141c28;
        --chopify-card-soft: #182230;
        --chopify-border: #2a3545;
        --chopify-muted: #a0aaba;
        --chopify-accent: #8b5cf6;
        --chopify-green: #4ade80;
    }

    html, body, .stApp, button, input, textarea,
    [data-baseweb="select"], [data-testid="stMarkdownContainer"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
            "Segoe UI Variable", "Segoe UI", sans-serif !important;
        text-rendering: optimizeLegibility;
        font-feature-settings: "kern" 1, "liga" 1, "cv02" 1, "cv03" 1;
        -webkit-font-smoothing: auto;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(circle at 58% -18%, rgba(92, 80, 154, 0.13), transparent 35rem),
            var(--chopify-bg);
        color: #f4f6fa;
        font-size: 15px;
        font-weight: 450;
        line-height: 1.5;
    }

    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    [data-testid="stToolbar"] { top: 0.3rem; }
    #MainMenu, footer, [data-testid="stAppDeployButton"] { visibility: hidden; }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1920px;
        width: 100%;
        padding: 1rem 1.1rem 1.15rem !important;
        margin-inline: auto;
    }

    [data-testid="stHorizontalBlock"] { gap: 0.85rem; }
    [data-testid="stVerticalBlock"] { gap: 0.62rem; }

    p, label, [data-testid="stCaptionContainer"] {
        letter-spacing: -0.006em;
        line-height: 1.48;
    }

    [data-testid="stCaptionContainer"] {
        color: #929dac;
        font-size: 0.82rem;
        font-weight: 480;
    }

    .nav-shell {
        min-height: 72vh;
        border: 1px solid var(--chopify-border);
        border-radius: 16px;
        background: #101016;
        padding: 1rem 0.75rem;
    }

    .nav-brand {
        font-size: 1.22rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -0.035em;
        color: #faf9ff;
        padding: 0.35rem 0.45rem 1.7rem;
        margin-bottom: 0.35rem;
        white-space: nowrap;
    }

    .nav-foot {
        margin-top: 55vh;
        padding: 0.75rem 0.35rem 0.1rem;
        border-top: 1px solid var(--chopify-border);
        color: #7f8a9a;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.66rem 0.7rem;
        margin-bottom: 0.25rem;
        border-radius: 10px;
        color: #9898a8;
        font-size: 0.86rem;
        font-weight: 650;
    }

    .nav-item.active {
        color: #ede9fe;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.24);
    }

    .nav-icon {
        width: 1.15rem;
        text-align: center;
        color: #a78bfa;
    }

    .product-header {
        margin-bottom: 1rem;
    }

    .product-name {
        font-size: 1.75rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
        color: #f5f3ff;
    }

    .product-mark { color: var(--chopify-accent); }

    .panel-heading {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.1rem 0 0.95rem;
        font-size: 1.18rem;
        line-height: 1.25;
        letter-spacing: -0.025em;
        font-weight: 750;
    }

    .panel-heading .spark { color: #a78bfa; }

    .eyebrow {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    h1, h2, h3 {
        color: #f7f8fb;
        letter-spacing: -0.035em;
        font-weight: 760;
        line-height: 1.16;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(66, 80, 100, 0.68);
        border-radius: 12px;
        background: linear-gradient(155deg, rgba(20, 28, 40, 0.98), rgba(13, 19, 29, 0.98));
        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-brand) {
        min-height: calc(100vh - 2rem);
        background: linear-gradient(180deg, rgba(11, 17, 27, 0.98), rgba(8, 13, 21, 0.98));
        padding: 0.95rem 0.72rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.new-project-panel) {
        min-height: calc(100vh - 2rem);
        padding: 1.15rem 1.15rem;
    }

    div[data-testid="stColumn"]:has(.new-project-panel)
    div[data-testid="stLayoutWrapper"]:has(.new-project-panel) {
        min-height: calc(100vh - 2rem);
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.clip-row-marker) {
        padding: 0.5rem 0.58rem;
        background: linear-gradient(180deg, #151e2a, #111923);
        border-color: rgba(61, 75, 94, 0.72);
        box-shadow: 0 7px 20px rgba(0, 0, 0, 0.1);
    }

    .clip-row-marker { display: none; }

    div[data-testid="stTextInput"] input {
        min-height: 3rem;
        border-radius: 9px;
        border: 1px solid #344155;
        color: #edf1f7;
        font-size: 0.91rem;
        font-weight: 500;
        background: #0b111b;
        box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.2);
    }

    div[data-baseweb="select"] > div {
        min-height: 3rem;
        border-radius: 9px;
        border-color: #344155;
        color: #edf1f7;
        background: #0b111b;
        font-size: 0.91rem;
        font-weight: 500;
        box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #d8dee8;
        font-size: 0.86rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--chopify-border);
        background: var(--chopify-card);
        padding: 0.55rem 0.7rem;
        border-radius: 9px;
    }

    div[data-testid="stExpander"] {
        border-color: #303c4d;
        border-radius: 9px;
        background: #121a25;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 9px;
        min-height: 2.75rem;
        font-weight: 650;
        font-size: 0.88rem;
        border-color: #354257;
        color: #e7ebf2;
        background: linear-gradient(180deg, #172130, #111925);
        box-shadow: 0 5px 14px rgba(0, 0, 0, 0.13);
        transition: transform 140ms ease, border-color 140ms ease,
            background 140ms ease, box-shadow 140ms ease;
    }

    div[data-testid="stButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px);
        border-color: #59677c;
        background: linear-gradient(180deg, #1c2838, #151e2b);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #995ff5, #7137dd);
        border-color: #9d69ed;
        color: white;
        box-shadow: 0 10px 25px rgba(112, 54, 219, 0.3), inset 0 1px 0 rgba(255,255,255,0.16);
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #a46cf7, #7d43e7);
        box-shadow: 0 13px 30px rgba(112, 54, 219, 0.38), inset 0 1px 0 rgba(255,255,255,0.2);
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-brand)
    div[data-testid="stButton"] button {
        justify-content: flex-start;
        min-height: 2.9rem;
        padding-inline: 0.8rem;
        border-color: transparent;
        background: transparent;
        box-shadow: none;
        font-size: 0.93rem;
        font-weight: 590;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-brand)
    div[data-testid="stButton"] button:hover {
        border-color: rgba(139, 92, 246, 0.18);
        background: rgba(139, 92, 246, 0.1);
        transform: none;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-brand)
    div[data-testid="stButton"] button[kind="primary"] {
        border-color: rgba(167, 139, 250, 0.3);
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.32), rgba(109, 65, 210, 0.2));
        box-shadow: inset 3px 0 0 #9d6cf5, 0 7px 18px rgba(65, 35, 125, 0.12);
    }

    .ai-status-card {
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 0.65rem;
        padding: 0.85rem;
        border: 1px solid #2c4a3a;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(25, 75, 50, 0.25), rgba(18, 42, 33, 0.17));
        margin: 0.3rem 0 0.9rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
    }

    .ai-status-card strong { color: #7aeca7; font-size: 0.88rem; font-weight: 680; }
    .ai-status-card small { display: block; color: #9ba6b5; font-size: 0.76rem; margin-top: 0.12rem; }
    .ai-check { color: var(--chopify-green); }

    .local-note {
        color: #8e99a9;
        text-align: center;
        font-size: 0.77rem;
        font-weight: 500;
        margin: 0.65rem 0 0;
    }

    .clip-title {
        color: #f4f6fa;
        font-size: 0.92rem;
        font-weight: 710;
        line-height: 1.3;
        letter-spacing: -0.018em;
        margin-bottom: 0.2rem;
    }

    .clip-type-badge {
        display: inline-block;
        margin: 0;
        padding: 0.16rem 0.52rem;
        border: 1px solid #4c3b76;
        border-radius: 999px;
        background: #251d38;
        color: #d0c3ff;
        font-size: 0.77rem;
        font-weight: 700;
    }

    .clip-why {
        margin: 0.1rem 0 0;
        color: #c0c8d4;
        font-size: 0.79rem;
        font-weight: 470;
        line-height: 1.38;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .clip-meta {
        color: var(--chopify-muted);
        color: #9ba7b7;
        font-size: 0.77rem;
        font-weight: 500;
        margin-bottom: 0.22rem;
    }

    .clip-preview {
        font-size: 0.79rem;
        font-weight: 450;
        line-height: 1.4;
        margin: 0;
        color: #d2d2dc;
        display: -webkit-box;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .thumb-placeholder {
        width: 100%;
        aspect-ratio: 16 / 9;
        border: 1px solid var(--chopify-border);
        border-radius: 8px;
        background: linear-gradient(135deg, #242433, #17171f);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #8b5cf6;
        font-size: 1.12rem;
    }

    .score-pill {
        display: inline-block;
        padding: 0.24rem 0.5rem;
        border-radius: 7px;
        border: 1px solid #354052;
        background: #1a2230;
        color: #cdd5df;
        font-size: 0.78rem;
        font-weight: 800;
        white-space: nowrap;
    }

    .score-pill.high { color: #78e39a; border-color: #27533a; background: #13291d; }
    .score-pill.medium { color: #f4ca55; border-color: #5b4d22; background: #29240f; }

    .mini-label {
        color: #98a4b4;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.16rem;
    }

    .workflow-row {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: #c8c8d2;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0.34rem 0;
    }

    .workflow-dot {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 999px;
        background: #4b4b58;
        flex: none;
    }

    .workflow-dot.active { background: var(--chopify-accent); }
    .workflow-dot.done { background: #34d399; }

    .empty-workspace {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(72, 84, 104, 0.62);
        background:
            radial-gradient(circle at 88% 12%, rgba(124, 83, 220, 0.14), transparent 24rem),
            linear-gradient(145deg, #151d29, #0f1621 72%);
        border-radius: 14px;
        padding: 2.25rem 2.35rem 2rem;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255,255,255,0.035);
    }

    .empty-workspace h1 {
        max-width: 760px;
        font-size: clamp(2.15rem, 3.2vw, 3.2rem);
        font-weight: 800;
        line-height: 1.06;
        letter-spacing: -0.045em;
        margin: 0 0 0.85rem 0;
    }

    .empty-workspace > p {
        color: #b0bac8 !important;
        font-size: 1rem;
        font-weight: 470;
        line-height: 1.65 !important;
    }

    .support-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.35rem;
    }

    .support-card {
        background: linear-gradient(155deg, rgba(26, 36, 51, 0.95), rgba(18, 26, 38, 0.95));
        border: 1px solid #303d4f;
        border-radius: 11px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12), inset 0 1px 0 rgba(255,255,255,0.025);
    }

    .support-card strong { display: block; font-size: 0.92rem; font-weight: 680; margin-bottom: 0.28rem; }
    .support-card span { color: #a2adbd; font-size: 0.84rem; line-height: 1.45; }

    .project-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0;
        padding: 0.82rem 1rem;
        border: 1px solid rgba(66, 80, 101, 0.7);
        border-radius: 11px;
        background: linear-gradient(145deg, #172130, #101721);
        box-shadow: 0 13px 32px rgba(0, 0, 0, 0.17), inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .workspace-label {
        color: #9ba6b6;
        font-size: 0.84rem;
        font-weight: 560;
        margin-bottom: 0.18rem;
    }

    .ready-pill {
        display: inline-block;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.28);
        font-size: 0.84rem;
        font-weight: 700;
    }

    .ai-pill {
        display: inline-block;
        margin-left: 0.4rem;
        padding: 0.32rem 0.65rem;
        border-radius: 999px;
        color: #b8c0ce;
        background: #161d28;
        border: 1px solid #2d3745;
        font-size: 0.8rem;
        font-weight: 650;
    }

    .project-title {
        margin: 0;
        color: #f4f6fa;
        font-size: 1.12rem;
        font-weight: 730;
        line-height: 1.32;
        letter-spacing: -0.025em;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0.5rem 0 0.25rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .count-pill {
        padding: 0.08rem 0.42rem;
        border-radius: 999px;
        background: #202938;
        color: #c5ccd6;
        font-size: 0.75rem;
        font-weight: 650;
    }

    .video-info {
        padding: 0.65rem 0.4rem;
    }

    .video-info-row {
        display: grid;
        grid-template-columns: 5.2rem 1fr;
        gap: 0.55rem;
        padding: 0.32rem 0;
        color: #d0d6df;
        font-size: 0.83rem;
        font-weight: 500;
    }

    .video-info-row span:first-child { color: #929dac; font-weight: 560; }

    .rendered-pill {
        display: inline-block;
        padding: 0.1rem 0.4rem;
        border-radius: 5px;
        background: #12321e;
        color: #61dc87;
        font-size: 0.75rem;
        font-weight: 700;
    }

    video {
        max-height: 360px;
        border-radius: 9px;
        background: #0c0c0c;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }

    hr {
        margin: 0.75rem 0 !important;
    }

    * { scrollbar-width: thin; scrollbar-color: #343e4c transparent; }

    @media (max-width: 1200px) and (min-width: 701px) {
        div[data-testid="stColumn"]:has(.nav-brand) {
            min-width: 168px;
            flex: 0 0 168px;
        }
        .nav-brand { font-size: 1.05rem; }
    }

    @media (max-width: 1100px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-brand),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.new-project-panel) {
            min-height: auto;
        }
        .nav-foot { margin-top: 1rem; }
        .support-grid { grid-template-columns: 1fr; }
        .empty-workspace { padding: 1.6rem; }
        .empty-workspace h1 { font-size: 2.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def score_clip(
    text,
    duration=40.0,
    speech_ratio=1.0,
    max_gap=0.0,
    return_breakdown=False
):
    """Score a candidate locally across five explainable quality factors."""
    lower = " ".join((text or "").lower().split())
    words = lower.split()
    hook_text = " ".join(words[:35])
    first_chars = lower[:180]

    question_starts = (
        "why ", "how ", "what ", "when ", "where ", "who ",
        "warum ", "wie ", "was ", "wann ", "wo ", "wer "
    )
    strong_hook_phrases = (
        "here's why", "here is why", "the problem is", "most people",
        "you need to", "the biggest mistake", "the truth is",
        "what nobody tells you", "this is why", "imagine this",
        "hier ist warum", "der grund ist", "das problem ist",
        "die meisten menschen", "die meisten leute", "du musst",
        "sie müssen", "der größte fehler", "der groesste fehler",
        "die wahrheit ist", "was dir niemand sagt", "stell dir vor"
    )
    assertion_words = (
        "never", "always", "must", "best", "worst", "impossible",
        "niemals", "immer", "muss", "beste", "schlimmste", "unmöglich"
    )

    hook = 0
    if "?" in first_chars or lower.startswith(question_starts):
        hook += 10
    hook += min(12, sum(6 for phrase in strong_hook_phrases if phrase in hook_text))
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", hook_text):
        hook += 5
    if any(re.search(rf"\b{re.escape(word)}\b", hook_text) for word in assertion_words):
        hook += 5
    if any(word in hook_text for word in ("but ", "however", "aber ", "doch ", "instead")):
        hook += 3
    hook = min(30, hook)

    context_openers = (
        "and then", "like i said", "as mentioned", "this one", "that ",
        "so yeah", "and so", "und dann", "wie gesagt", "wie erwähnt",
        "wie erwaehnt", "dieses hier", "das hier", "also ja", "und so"
    )
    vague_openers = (
        "and ", "so ", "then ", "because ", "that ", "it ",
        "und ", "also ", "dann ", "weil ", "das ", "es "
    )
    clarity = 20
    if lower.startswith(context_openers):
        clarity -= 12
    elif lower.startswith(vague_openers):
        clarity -= 6
    if len(words) < 35:
        clarity -= 5
    if text.rstrip().endswith((".", "!", "?")):
        clarity += 2
    clarity = max(0, min(20, clarity))

    words_per_second = len(words) / max(duration, 1.0)
    if 1.6 <= words_per_second <= 3.4:
        density = 20
    elif 1.2 <= words_per_second < 1.6 or 3.4 < words_per_second <= 4.0:
        density = 15
    elif 0.85 <= words_per_second < 1.2 or 4.0 < words_per_second <= 4.6:
        density = 9
    else:
        density = 4
    if speech_ratio < 0.72:
        density -= 6
    elif speech_ratio < 0.84:
        density -= 3
    if max_gap > 4.0:
        density -= 6
    elif max_gap > 2.5:
        density -= 3
    density = max(0, min(20, density))

    novelty_terms = (
        "surprise", "surprising", "shock", "shocking", "secret", "crazy",
        "insane", "love", "hate", "fear", "angry", "risk", "mistake",
        "problem", "truth", "unexpected", "überrasch", "ueberrasch",
        "schock", "geheimnis", "verrückt", "verrueckt", "liebe", "hasse",
        "angst", "wütend", "wuetend", "risiko", "fehler", "wahrheit"
    )
    novelty_hits = sum(1 for term in novelty_terms if term in lower)
    novelty = min(12, novelty_hits * 3)
    if "!" in text:
        novelty += 2
    if any(term in lower for term in ("but", "however", "instead", "aber", "doch", "stattdessen")):
        novelty += 2
    novelty = min(15, novelty)

    if 32 <= duration <= 48:
        length_quality = 15
    elif 25 <= duration <= 55:
        length_quality = 12
    elif 20 <= duration <= 60:
        length_quality = 6
    else:
        length_quality = 0

    breakdown = {
        "hook": hook,
        "standalone_clarity": clarity,
        "information_density": density,
        "emotion_novelty": novelty,
        "length_quality": length_quality
    }
    score = max(0, min(100, sum(breakdown.values())))
    return (score, breakdown) if return_breakdown else score


def build_clip_candidates(segments):
    """Build one natural-length candidate per selective start anchor, then dedupe."""
    clean_segments = []
    for segment in segments:
        start = float(segment.get("start", 0) or 0)
        end = float(segment.get("end", 0) or 0)
        text = segment.get("text", "").strip()
        if text and end > start:
            clean_segments.append({"start": start, "end": end, "text": text})

    if not clean_segments:
        return []

    anchor_phrases = (
        "here's why", "here is why", "the problem is", "most people",
        "you need to", "the biggest mistake", "the truth is",
        "hier ist warum", "der grund ist", "das problem ist",
        "die meisten menschen", "die meisten leute", "du musst",
        "der größte fehler", "der groesste fehler", "die wahrheit ist"
    )
    question_words = (
        "why ", "how ", "what ", "when ", "who ",
        "warum ", "wie ", "was ", "wann ", "wer "
    )

    anchors = [0]
    last_anchor_time = clean_segments[0]["start"]
    for index in range(1, len(clean_segments)):
        segment = clean_segments[index]
        lower = segment["text"].lower().lstrip()
        previous = clean_segments[index - 1]
        gap = max(0.0, segment["start"] - previous["end"])
        natural_start = previous["text"].rstrip().endswith((".", "!", "?")) or gap >= 0.65
        strong_start = (
            "?" in segment["text"][:180]
            or lower.startswith(question_words)
            or any(phrase in lower[:220] for phrase in anchor_phrases)
        )
        since_last = segment["start"] - last_anchor_time

        if strong_start or (since_last >= 18 and natural_start) or since_last >= 30:
            anchors.append(index)
            last_anchor_time = segment["start"]

    candidates = []
    for start_index in anchors:
        start = clean_segments[start_index]["start"]
        possible_ends = []

        for end_index in range(start_index, len(clean_segments)):
            end = clean_segments[end_index]["end"]
            duration = end - start
            if duration > 55:
                break
            if duration < 25:
                continue

            next_gap = 0.0
            if end_index + 1 < len(clean_segments):
                next_gap = max(0.0, clean_segments[end_index + 1]["start"] - end)
            terminal = clean_segments[end_index]["text"].rstrip().endswith((".", "!", "?"))
            boundary_quality = (
                8.0 - abs(duration - 40.0) / 4.0
                + (4.0 if terminal else 0.0)
                + min(3.0, next_gap * 2.0)
            )
            possible_ends.append((boundary_quality, end_index))

        if not possible_ends:
            continue

        _, end_index = max(possible_ends, key=lambda item: item[0])
        chosen = clean_segments[start_index:end_index + 1]
        end = chosen[-1]["end"]
        duration = end - start
        text = " ".join(segment["text"] for segment in chosen).strip()
        spoken_duration = sum(segment["end"] - segment["start"] for segment in chosen)
        gaps = [
            max(0.0, chosen[i + 1]["start"] - chosen[i]["end"])
            for i in range(len(chosen) - 1)
        ]
        score, breakdown = score_clip(
            text,
            duration=duration,
            speech_ratio=spoken_duration / max(duration, 1.0),
            max_gap=max(gaps, default=0.0),
            return_breakdown=True
        )
        candidates.append({
            "start": start,
            "end": end,
            "duration": duration,
            "text": text,
            "score": score,
            "score_breakdown": breakdown
        })

    candidates.sort(key=lambda clip: clip["score"], reverse=True)
    deduplicated = []
    for candidate in candidates:
        duplicate = False
        for kept in deduplicated:
            overlap = max(
                0.0,
                min(candidate["end"], kept["end"])
                - max(candidate["start"], kept["start"])
            )
            overlap_ratio = overlap / min(candidate["duration"], kept["duration"])
            if overlap_ratio > 0.60:
                duplicate = True
                break

        if not duplicate:
            deduplicated.append(candidate)
        if len(deduplicated) >= 15:
            break

    return deduplicated


def build_hook_options(clip, words):
    """Return a few strong, word-aligned cold-open options from inside one clip."""
    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    timed_words = [
        word for word in words
        if float(word.get("end", 0)) > clip_start
        and float(word.get("start", 0)) < clip_end
        and str(word.get("word", "")).strip()
    ]
    if not timed_words:
        return []

    sentence_starts = [0]
    for index, word in enumerate(timed_words[:-1]):
        if str(word["word"]).rstrip().endswith((".", "!", "?", "。", "！", "？")):
            sentence_starts.append(index + 1)

    strong_phrases = (
        "the truth", "the problem", "the biggest", "nobody", "never", "always",
        "i lost", "i made", "here's why", "this is why", "most people",
        "die wahrheit", "das problem", "der größte", "der groesste", "niemand",
        "niemals", "immer", "ich verlor", "ich habe verloren", "deshalb",
    )
    emotion_terms = (
        "lost", "won", "hate", "love", "fear", "shocked", "crazy", "insane",
        "mistake", "risk", "verloren", "gewonnen", "hasse", "liebe", "angst",
        "schock", "verrückt", "verrueckt", "fehler", "risiko",
    )
    weak_starts = (
        "and ", "so ", "but ", "then ", "like ", "um ", "uh ", "well ",
        "und ", "also ", "aber ", "dann ", "äh ", "aeh ", "naja ",
    )

    options = []
    for start_index in sentence_starts:
        source_start = max(clip_start, float(timed_words[start_index]["start"]))
        # Repeating the opening itself adds no useful cold-open structure.
        if source_start - clip_start < 1.0:
            continue

        for end_index in range(start_index, min(len(timed_words), start_index + 18)):
            source_end = min(clip_end, float(timed_words[end_index]["end"]))
            duration = source_end - source_start
            if duration < 1.35:
                continue
            if duration > 3.5:
                break
            terminal = str(timed_words[end_index]["word"]).rstrip().endswith(
                (".", "!", "?", "。", "！", "？")
            )
            if not terminal and duration < 2.5:
                continue

            text = " ".join(
                str(word["word"]).strip()
                for word in timed_words[start_index:end_index + 1]
            ).strip()
            text = re.sub(r"\s+([,.!?;:])", r"\1", text)
            normalized = " ".join(text.lower().split())
            if not text or normalized.startswith(weak_starts):
                continue

            score = 0
            if re.search(r"\b\d+(?:[.,]\d+)?(?:%|k|m|€|\$)?\b", normalized):
                score += 9
            score += min(8, sum(4 for phrase in strong_phrases if phrase in normalized))
            score += min(6, sum(3 for term in emotion_terms if term in normalized))
            if "?" in text:
                score += 4
            if "!" in text:
                score += 3
            if terminal:
                score += 5
            word_count = len(normalized.split())
            if 4 <= word_count <= 14:
                score += 4
            if 1.5 <= duration <= 2.5:
                score += 4

            options.append({
                "text": text,
                "start": round(source_start, 3),
                "end": round(source_end, 3),
                "score": score,
            })
            if terminal:
                break

    options.sort(key=lambda option: option["score"], reverse=True)
    return options[:5]


def apply_local_hook_defaults(candidates, enabled=True):
    """Set a hook only when the best local option clears a conservative threshold."""
    prepared = []
    for candidate in candidates:
        clip = dict(candidate)
        if not enabled:
            clip.update({
                "automatic_hook": False,
                "hook_text": "",
                "hook_start": 0.0,
                "hook_end": 0.0,
                "hook_reason": "",
                "selected_hook_option_id": None,
            })
        elif "automatic_hook" not in clip:
            options = clip.get("_hook_options", [])
            best = options[0] if options else None
            enabled = bool(best and int(best["score"]) >= 14)
            clip.update({
                "automatic_hook": enabled,
                "hook_text": best["text"] if enabled else "",
                "hook_start": best["start"] if enabled else 0.0,
                "hook_end": best["end"] if enabled else 0.0,
                "hook_reason": (
                    "Strong standalone statement selected from the clip."
                    if enabled else ""
                ),
            })
        prepared.append(clip)
    return prepared


def campaign_context_for_ai(campaign):
    if not isinstance(campaign, dict):
        return None
    context = {
        "campaign_name": str(campaign.get("campaign_name") or "")[:120],
        "target_platforms": campaign.get("target_platforms") or [],
        "content_language": str(campaign.get("content_language") or "")[:40],
        "required_hashtags": campaign.get("required_hashtags") or [],
        "required_mentions": campaign.get("required_mentions") or [],
        "notes": str(campaign.get("notes") or "")[:600],
    }
    return context if any(context.values()) else None


def estimate_campaign_fit(candidate, campaign):
    context = campaign_context_for_ai(campaign)
    if not context:
        return 0
    campaign_text = " ".join(
        [context["campaign_name"], context["content_language"], context["notes"]]
        + [str(value) for key in ("target_platforms", "required_hashtags", "required_mentions")
           for value in context[key]]
    ).lower()
    clip_text = str(candidate.get("text") or "").lower()
    campaign_tokens = {
        token for token in re.findall(r"\w+", campaign_text) if len(token) >= 4
    }
    clip_tokens = set(re.findall(r"\w+", clip_text))
    overlap = len(campaign_tokens & clip_tokens)
    phrase_bonus = sum(
        phrase in campaign_text and phrase in clip_text
        for phrase in (
            "trading", "crypto", "founder", "loss", "losses", "win", "money",
            "million", "viral", "story", "hot take", "fomo", "investment",
        )
    )
    return min(10, overlap * 2 + phrase_bonus * 2)


def ai_candidate_payload(candidates, campaign=None, ranking_options=None):
    ranking_options = normalize_ranking_options(ranking_options)
    candidate_payload = [
        {
            "candidate_id": f"candidate_{index}",
            "start": round(float(clip["start"]), 2),
            "end": round(float(clip["end"]), 2),
            "duration": round(float(clip["duration"]), 2),
            "transcript": clip["text"],
            "audio": clip.get("audio", {}),
            "visual": clip.get("visual", {}),
            "source_position": {
                "seconds": round(float(clip["start"]), 2),
                "relative": round(float(clip.get("source_position", 0.0)), 4),
            },
            "region": {
                "region_id": clip.get("region_id"),
                "local_interest_score": round(float(clip.get("local_interest_score", 0)), 4),
                "region_score": round(float(clip.get("region_score", 0)), 4),
                "rank": clip.get("region_rank"),
                "variant_type": clip.get("variant_type", "STANDARD"),
                "anchor_types": clip.get("anchor_types", []),
                "anchor_strengths": clip.get("anchor_strengths", {}),
                "multimodal_signal_count": sum((
                    bool(clip.get("text")),
                    any(float(value or 0) > 0 for value in clip.get("audio", {}).values()
                        if isinstance(value, (int, float))),
                    any(float(value or 0) > 0 for value in clip.get("visual", {}).values()
                        if isinstance(value, (int, float))),
                )),
            },
            "dead_air_ratio": round(float(clip.get("dead_air_ratio", 0)), 4),
            "surrounding_context": clip.get("surrounding_context", {}),
            "local_campaign_fit_hint": estimate_campaign_fit(clip, campaign),
            "hook_options": [
                {
                    "id": option_index,
                    "text": option["text"],
                    "start": round(float(option["start"]), 3),
                    "end": round(float(option["end"]), 3),
                    "local_strength": int(option["score"]),
                }
                for option_index, option in enumerate(
                    (
                        clip.get("_hook_options", [])
                        if ranking_options["two_second_hook"]
                        else []
                    ),
                    start=1,
                )
            ],
        }
        for index, clip in enumerate(candidates, start=1)
    ]
    return {
        "ranking_options": {
            "viral_shorts": ranking_options["viral_shorts"],
            "two_second_hook": ranking_options["two_second_hook"],
        },
        "campaign": campaign_context_for_ai(campaign),
        "candidates": candidate_payload,
    }


def ai_scoring_schema(candidate_count):
    evaluation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {
                "type": "string",
                "enum": [f"candidate_{index}" for index in range(1, candidate_count + 1)],
            },
            "decision": {
                "type": "string",
                "enum": ["KEEP", "REJECT"],
                "description": "Publishability gate. Reject freely; there is no target clip count.",
            },
            "title": {
                "type": "string", "maxLength": 80,
                "description": "Natural specific title in the spoken language; maximum 8 words.",
            },
            "scroll_stop_score": {
                "type": "integer", "minimum": 0, "maximum": 25,
                "description": "Scroll-stop power, pattern interrupt, and immediate curiosity or impact.",
            },
            "retention_score": {
                "type": "integer", "minimum": 0, "maximum": 20,
                "description": "Likelihood viewers keep watching instead of swiping away.",
            },
            "payoff_score": {
                "type": "integer", "minimum": 0, "maximum": 15,
                "description": "Strength and completeness of reveal, answer, punchline, or story payoff.",
            },
            "emotion_novelty_score": {
                "type": "integer", "minimum": 0, "maximum": 15,
                "description": "Emotion, surprise, novelty, tension, conflict, humor, or absurdity.",
            },
            "standalone_score": {
                "type": "integer", "minimum": 0, "maximum": 10,
                "description": "Context-free clarity as a standalone short-form clip.",
            },
            "shareability_score": {
                "type": "integer", "minimum": 0, "maximum": 10,
                "description": "Quotability and likelihood viewers share or discuss the moment.",
            },
            "pacing_score": {
                "type": "integer", "minimum": 0, "maximum": 5,
                "description": "Information density, efficient length, and lack of dead air or repetition.",
            },
            "natural_hook_quality": {
                "type": "integer", "minimum": 0, "maximum": 10,
                "description": "Quality of the candidate's unmodified first 1-2 seconds.",
            },
            "hook_improved_quality": {
                "type": "integer", "minimum": 0, "maximum": 10,
                "description": "Opening quality with the selected hook; equal natural quality when none is selected.",
            },
            "viral_reason": {
                "type": "string", "maxLength": 150,
                "description": "Viral upside in the spoken language; maximum 18 words.",
            },
            "weakness": {
                "type": "string", "maxLength": 100,
                "description": "Main weakness in the spoken language; maximum 12 words.",
            },
            "reject_reason": {
                "type": "string", "maxLength": 140,
                "description": "Concrete reason for REJECT in at most 16 words; empty for KEEP.",
            },
            "clip_type": {
                "type": "string",
                "enum": [
                    "Big Number", "Wild Story", "Hot Take", "Conflict", "Reaction",
                    "Funny", "Reveal", "Emotional", "Educational", "Prediction",
                    "Founder Story", "Debate", "Unexpected", "Other",
                ],
            },
            "selected_hook_option_id": {
                "anyOf": [
                    {"type": "integer", "minimum": 1, "maximum": 5},
                    {"type": "null"},
                ],
                "description": "ID of one supplied hook option, or null when none is strong.",
            },
            "relative_rank": {
                "type": "integer",
                "minimum": 1,
                "maximum": candidate_count,
                "description": "Position among ALL candidates in this video (1 = best). Unique per candidate.",
            },
            "campaign_fit_score": {
                "type": "integer", "minimum": 0, "maximum": 10,
                "description": "Campaign relevance only; return 0 when no campaign context is supplied.",
            },
        },
        "required": [
            "candidate_id", "decision", "title", "scroll_stop_score",
            "retention_score", "payoff_score", "emotion_novelty_score",
            "standalone_score", "shareability_score", "pacing_score",
            "natural_hook_quality", "hook_improved_quality", "viral_reason",
            "weakness", "reject_reason", "clip_type", "selected_hook_option_id",
            "relative_rank", "campaign_fit_score",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evaluations": {
                "type": "array",
                "items": evaluation,
                "minItems": candidate_count,
                "maxItems": candidate_count,
            }
        },
        "required": ["evaluations"],
    }


def quality_tier_for_score(score):
    return calibrated_quality_tier_for_score(score)


def candidate_evidence_summary(candidate, components):
    text = " ".join(str(candidate.get("text") or "").lower().split())
    audio = candidate.get("audio") if isinstance(candidate.get("audio"), dict) else {}
    visual = candidate.get("visual") if isinstance(candidate.get("visual"), dict) else {}
    anchor_types = {str(value).lower() for value in candidate.get("anchor_types", [])}
    local_interest = float(candidate.get("local_interest_score", 0) or 0)
    strong_language = bool(re.search(
        r"(?:\b\d+(?:[.,]\d+)?(?:%|k|m|million|billion)?\b|\$|€|"
        r"top.?secret|the truth|the biggest|the problem|never|impossible|"
        r"conflict|shocking|revea|die wahrheit|das problem|niemals|unmöglich|"
        r"überrasch|enthüll)",
        text,
    ))
    strong_text_event = bool(
        ("text" in anchor_types and local_interest >= 0.52)
        or (strong_language and local_interest >= 0.62)
    )
    strong_audio_event = bool(
        float(audio.get("peak_energy", 0) or 0) >= 0.78
        and (
            int(audio.get("energy_spikes", 0) or 0) >= 2
            or int(audio.get("rapid_speech_events", 0) or 0) >= 2
            or int(audio.get("laughter_like_events", 0) or 0) >= 1
        )
    )
    # Motion by itself is deliberately insufficient: it may only be camera movement.
    strong_visual_event = bool(
        (
            int(visual.get("scene_changes", 0) or 0) >= 2
            and float(visual.get("visual_change_peak", 0) or 0) >= 0.72
        )
        or int(visual.get("face_count_changes", 0) or 0) >= 2
    )
    strong_novelty_conflict = bool(
        int(components.get("emotion_novelty", 0)) >= 12 and strong_language
    )
    strong_shareability = bool(
        int(components.get("shareability", 0)) >= 8
        and (
            strong_language
            or re.search(
                r"\b(?:he|she|they|elon|musk|everyone|nobody|people|"
                r"er|sie|niemand|alle|menschen)\b.{0,80}\b(?:is|was|has|bought|"
                r"said|owns|ist|war|hat|kaufte|besitzt)\b",
                text,
            )
        )
    )
    evidence = {
        "strong_text_event": strong_text_event,
        "strong_audio_event": strong_audio_event,
        "strong_visual_event": strong_visual_event,
        "strong_novelty_conflict_evidence": strong_novelty_conflict,
        "strong_shareability_evidence": strong_shareability,
    }
    source_groups = []
    if strong_text_event or strong_novelty_conflict or strong_shareability:
        source_groups.append("transcript")
    if strong_audio_event:
        source_groups.append("audio")
    if strong_visual_event:
        source_groups.append("visual_change_metrics")
    evidence["strong_evidence_count"] = sum(bool(value) for value in evidence.values())
    evidence["independent_source_groups"] = source_groups
    evidence["transcript_dominant"] = bool(
        "transcript" in source_groups
        and "audio" not in source_groups
        and "visual_change_metrics" not in source_groups
    )
    evidence["visual_semantics_available"] = any(
        key in visual
        for key in ("objects", "detected_objects", "ocr_text", "room_type", "reaction_labels")
    )
    return evidence


def candidate_confidence(candidate, evidence):
    text = str(candidate.get("text") or "").strip()
    audio = candidate.get("audio") if isinstance(candidate.get("audio"), dict) else {}
    visual = candidate.get("visual") if isinstance(candidate.get("visual"), dict) else {}
    context = (
        candidate.get("surrounding_context")
        if isinstance(candidate.get("surrounding_context"), dict)
        else {}
    )
    word_count = len(text.split())
    breakdown = {
        "transcript": round(min(0.30, word_count / 80 * 0.30), 3),
        "audio_metrics": 0.20 if any(
            key in audio for key in ("peak_energy", "energy_spikes", "rapid_speech_events")
        ) else 0.0,
        # Numeric motion/change coverage helps confidence, but cannot provide visual meaning.
        "visual_change_metrics": 0.10 if any(
            key in visual for key in ("scene_changes", "motion_peak", "visual_change_peak")
        ) else 0.0,
        "visual_semantics": 0.15 if evidence.get("visual_semantics_available") else 0.0,
        "surrounding_context": 0.10 if context.get("before") or context.get("after") else 0.0,
        "region_metadata": 0.08 if candidate.get("region_id") is not None else 0.0,
        "hook_options": 0.03 if candidate.get("_hook_options") else 0.0,
        "cross_modal_support": min(
            0.04,
            max(0, len(evidence.get("independent_source_groups", [])) - 1) * 0.02,
        ),
    }
    return round(min(1.0, sum(breakdown.values())), 3), breakdown


def apply_score_confidence_caps(candidate, components):
    return calibrated_score_confidence_caps(
        candidate,
        components,
        evidence_fn=candidate_evidence_summary,
        confidence_fn=candidate_confidence,
    )


def apply_quality_guardrails(candidate, components):
    return calibrated_quality_guardrails(candidate, components)


def apply_ai_evaluations(candidates, evaluations, allow_synthetic_hook=True):
    expected = {f"candidate_{index}" for index in range(1, len(candidates) + 1)}
    by_id = {}
    for evaluation in evaluations:
        candidate_id = evaluation.get("candidate_id")
        if candidate_id not in expected or candidate_id in by_id:
            raise ValueError("AI response contains invalid or duplicate candidate IDs")
        by_id[candidate_id] = evaluation
    if set(by_id) != expected:
        raise ValueError("AI response does not cover every candidate")

    keep_ranks = []
    ranked = []
    for index, candidate in enumerate(candidates, start=1):
        evaluation = by_id[f"candidate_{index}"]
        score_ranges = {
            "scroll_stop_score": (0, 25),
            "retention_score": (0, 20),
            "payoff_score": (0, 15),
            "emotion_novelty_score": (0, 15),
            "standalone_score": (0, 10),
            "shareability_score": (0, 10),
            "pacing_score": (0, 5),
            "natural_hook_quality": (0, 10),
            "hook_improved_quality": (0, 10),
        }
        for field, (minimum, maximum) in score_ranges.items():
            value = int(evaluation[field])
            if not minimum <= value <= maximum:
                raise ValueError(f"AI score outside allowed range: {field}")
        model_decision = evaluation["decision"]
        model_relative_rank = evaluation.get("relative_rank")
        if model_relative_rank is None:
            raise ValueError("AI candidate is missing relative_rank among the full set")
        model_relative_rank = int(model_relative_rank)
        if not 1 <= model_relative_rank <= len(candidates):
            raise ValueError("AI relative_rank outside candidate set bounds")
        keep_ranks.append(model_relative_rank)
        if model_decision == "KEEP":
            if any(
                not evaluation[field].strip()
                for field in ("title", "viral_reason", "weakness")
            ):
                raise ValueError("AI KEEP candidate contains an empty text field")
        if model_decision == "REJECT" and not evaluation["reject_reason"].strip():
            raise ValueError("AI REJECT candidate has no reject_reason")
        components = {
            "hook": int(evaluation["scroll_stop_score"]),
            "retention": int(evaluation["retention_score"]),
            "payoff": int(evaluation["payoff_score"]),
            "emotion_novelty": int(evaluation["emotion_novelty_score"]),
            "standalone": int(evaluation["standalone_score"]),
            "shareability": int(evaluation["shareability_score"]),
            "pacing": int(evaluation["pacing_score"]),
        }
        gpt_components = dict(components)
        components, penalties = apply_quality_guardrails(candidate, components)
        score_details = apply_score_confidence_caps(candidate, components)
        raw_component_total = score_details["raw_score"]
        component_total = score_details["final_score"]
        selected_hook = None
        hook_option_id = evaluation.get("selected_hook_option_id")
        if hook_option_id is not None and allow_synthetic_hook:
            options = candidate.get("_hook_options", [])
            if not isinstance(hook_option_id, int) or not 1 <= hook_option_id <= len(options):
                raise ValueError("AI response selected an invalid hook_option_id")
            selected_hook = options[hook_option_id - 1]

        natural_hook_quality = int(evaluation["natural_hook_quality"])
        hook_improved_quality = int(evaluation["hook_improved_quality"])
        if selected_hook is None:
            hook_improved_quality = natural_hook_quality

        severe_gate_failures = []
        if components["payoff"] <= 4:
            severe_gate_failures.append("no credible payoff")
        if components["standalone"] <= 3:
            severe_gate_failures.append("requires missing context")
        if components["hook"] <= 8 and not (
            selected_hook is not None and hook_improved_quality >= 6
        ):
            severe_gate_failures.append("opening is too weak")
        if components["retention"] <= 6 and components["pacing"] <= 1:
            severe_gate_failures.append("insufficient forward momentum")
        non_hook_substance = sum(
            components[field]
            for field in (
                "retention", "payoff", "emotion_novelty",
                "standalone", "shareability", "pacing",
            )
        )
        if selected_hook is not None and non_hook_substance < 36:
            severe_gate_failures.append("auto-hook cannot rescue weak underlying content")

        narrow_gate_failures = []
        if components["hook"] == 9 and selected_hook is None:
            narrow_gate_failures.append("opening narrowly misses the hook gate")
        if components["standalone"] == 4:
            narrow_gate_failures.append("standalone context is borderline")
        if components["retention"] == 7 and components["pacing"] <= 2:
            narrow_gate_failures.append("forward momentum is borderline")

        decision = decide_publishability(
            model_decision=model_decision,
            component_total_score=component_total,
            severe_gate_failures=severe_gate_failures,
            narrow_gate_failures=narrow_gate_failures,
        )

        if decision == "REJECT":
            selected_hook = None
            hook_option_id = None
        # relative_rank is among ALL candidates; finalized after the full batch.
        final_relative_rank = int(model_relative_rank)

        def compact_text(value, maximum_words):
            return " ".join(str(value).strip().split()[:maximum_words])

        ranked_clip = dict(candidate)
        ranked_clip.update({
            "candidate_id": f"candidate_{index}",
            "local_score": candidate["score"],
            "score": component_total,
            "decision": decision,
            "model_decision": model_decision,
            "relative_rank": final_relative_rank,
            "model_relative_rank": model_relative_rank,
            "reject_reason": compact_text(
                "; ".join(
                    filter(None, [
                        evaluation["reject_reason"].strip(),
                        *severe_gate_failures,
                        *narrow_gate_failures,
                        (
                            "below KEEP threshold"
                            if decision == "BORDERLINE" and component_total < 75
                            else ""
                        ),
                        (
                            "overall score below review range"
                            if decision == "REJECT" and component_total < 68
                            else ""
                        ),
                    ])
                ),
                22,
            ) if decision != "KEEP" else "",
            "title": compact_text(evaluation["title"], 8),
            "ai_reason": compact_text(evaluation["viral_reason"], 18),
            "viral_reason": compact_text(evaluation["viral_reason"], 18),
            "ai_weakness": compact_text(evaluation["weakness"], 12),
            "clip_type": evaluation["clip_type"],
            "ai_scores": {
                "hook": components["hook"],
                "retention": components["retention"],
                "payoff": components["payoff"],
                "emotion_novelty": components["emotion_novelty"],
                "emotion_novelty_conflict": components["emotion_novelty"],
                "standalone": components["standalone"],
                "shareability": components["shareability"],
                "pacing": components["pacing"],
            },
            "gpt_component_scores": {
                "scroll_stop": gpt_components["hook"],
                "retention": gpt_components["retention"],
                "payoff": gpt_components["payoff"],
                "emotion_novelty_conflict": gpt_components["emotion_novelty"],
                "standalone": gpt_components["standalone"],
                "shareability": gpt_components["shareability"],
                "pacing": gpt_components["pacing"],
            },
            "quality_tier": quality_tier_for_score(component_total),
            "campaign_fit_score": int(evaluation["campaign_fit_score"]),
            "penalties": penalties,
            "natural_hook_quality": natural_hook_quality,
            "hook_improved_quality": hook_improved_quality,
            "raw_score": raw_component_total,
            "score_cap": score_details["score_cap"],
            "score_cap_reasons": score_details["score_cap_reasons"],
            "confidence_score": score_details["confidence_score"],
            "confidence_breakdown": score_details["confidence_breakdown"],
            "evidence": score_details["evidence"],
            "quality_gate_failures": severe_gate_failures,
            "borderline_signals": narrow_gate_failures,
            "selected_hook_option_id": hook_option_id,
            "scoring_mode": "ai",
            "automatic_hook": selected_hook is not None,
            "hook_text": selected_hook["text"] if selected_hook else "",
            "hook_start": selected_hook["start"] if selected_hook else 0.0,
            "hook_end": selected_hook["end"] if selected_hook else 0.0,
            "hook_reason": (
                "AI selected this local hook option." if selected_hook else ""
            ),
        })
        ranked.append(ranked_clip)
    if len(keep_ranks) != len(set(keep_ranks)):
        raise ValueError("AI candidates contain duplicate relative_rank values")
    if set(keep_ranks) != set(range(1, len(candidates) + 1)):
        raise ValueError("AI relative_rank values must cover every position in the set")
    return finalize_relative_ranks(ranked)


def _semantic_vectors(candidates):
    stopwords = {
        "the", "and", "that", "this", "with", "from", "have", "was", "for",
        "you", "your", "but", "are", "not", "und", "der", "die", "das", "ist",
        "mit", "von", "für", "aber", "nicht", "ich", "wir", "sie", "ein", "eine",
    }
    documents = []
    document_frequency = {}
    for candidate in candidates:
        tokens = [
            token for token in re.findall(r"\w+", str(candidate.get("text", "")).lower())
            if len(token) >= 3 and token not in stopwords
        ]
        documents.append(tokens)
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    count = max(1, len(documents))
    vectors = []
    for tokens in documents:
        frequencies = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        vector = {
            token: frequency * (math.log((count + 1) / (document_frequency[token] + 1)) + 1)
            for token, frequency in frequencies.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({token: value / norm for token, value in vector.items()})
    return vectors


def _cosine_similarity(first, second):
    if len(first) > len(second):
        first, second = second, first
    return sum(value * second.get(token, 0.0) for token, value in first.items())


def _token_containment_similarity(first_text, second_text):
    def tokens(value):
        return {
            token for token in re.findall(r"\w+", str(value).lower())
            if len(token) >= 3
        }
    first = tokens(first_text)
    second = tokens(second_text)
    return len(first & second) / max(1, min(len(first), len(second)))


def candidate_ranking_record(clip, display_rank=None):
    return {
        "candidate_id": clip.get("candidate_id"),
        "candidate_source": clip.get("candidate_source", "multimodal"),
        "rank": display_rank,
        "display_rank": display_rank,
        "relative_rank": clip.get("relative_rank"),
        "decision": clip.get("decision", "KEEP"),
        "model_decision": clip.get("model_decision"),
        "model_relative_rank": clip.get("model_relative_rank"),
        "reject_reason": clip.get("reject_reason", ""),
        "start": round(float(clip["start"]), 3),
        "end": round(float(clip["end"]), 3),
        "score": int(clip["score"]),
        "raw_score": int(clip.get("raw_score", clip["score"])),
        "score_cap": int(clip.get("score_cap", 100)),
        "score_cap_reasons": clip.get("score_cap_reasons", []),
        "gpt_component_scores": clip.get("gpt_component_scores", {}),
        "component_scores": clip.get("ai_scores", {}),
        "confidence_score": float(clip.get("confidence_score", 0.0)),
        "confidence_breakdown": clip.get("confidence_breakdown", {}),
        "evidence": clip.get("evidence", {}),
        "quality_gate_failures": clip.get("quality_gate_failures", []),
        "borderline_signals": clip.get("borderline_signals", []),
        "natural_hook_quality": int(clip.get("natural_hook_quality", 0)),
        "hook_improved_quality": int(clip.get("hook_improved_quality", 0)),
        "selected_hook_option_id": clip.get("selected_hook_option_id"),
        "quality_tier": clip.get("quality_tier"),
        "campaign_fit_score": int(clip.get("campaign_fit_score", 0)),
        "title": clip.get("title"),
        "clip_type": clip.get("clip_type"),
        "viral_reason": clip.get("viral_reason"),
        "weakness": clip.get("ai_weakness"),
        "penalties": clip.get("penalties", []),
        "region_id": clip.get("region_id"),
        "variant_type": clip.get("variant_type"),
        "variant_winner": bool(clip.get("variant_winner")),
        "local_score": int(clip.get("local_score", clip["score"])),
        "automatic_hook": bool(clip.get("automatic_hook")),
        "hook_start": float(clip.get("hook_start") or 0),
        "hook_end": float(clip.get("hook_end") or 0),
        "hook_text": clip.get("hook_text", ""),
        "hook_reason": clip.get("hook_reason", ""),
    }


def postprocess_ranked_candidates(ranked, campaign=None):
    removed = []
    keep_candidates = []
    review_candidates = []
    all_candidates = []
    for index, candidate in enumerate(ranked, start=1):
        clip = dict(candidate)
        clip.setdefault("candidate_id", f"candidate_{index}")
        clip.setdefault("quality_tier", quality_tier_for_score(int(clip.get("score", 0))))
        clip.setdefault("campaign_fit_score", estimate_campaign_fit(clip, campaign))
        clip.setdefault("decision", "KEEP")
        clip["pre_diversity_rank"] = index
        all_candidates.append(clip)
        if clip["decision"] == "KEEP":
            keep_candidates.append(clip)
        else:
            removed.append({
                "candidate": clip,
                "removed_reason": (
                    "borderline_quality_gate"
                    if clip["decision"] == "BORDERLINE"
                    else "ai_quality_gate"
                ),
            })
            if int(clip.get("score", 0)) >= EXPERIMENTAL_REVIEW_SCORE_FLOOR:
                review_candidates.append(clip)
    keep_candidates.sort(key=lambda clip: (
        clip.get("relative_rank") is None,
        clip.get("relative_rank") or len(ranked) + 1,
        -int(clip.get("score", 0)),
    ))

    region_winners = []
    seen_regions = {}
    for clip in keep_candidates:
        region_id = clip.get("region_id")
        if region_id is None:
            region_winners.append(clip)
            continue
        if region_id in seen_regions:
            removed.append({
                "candidate": clip,
                "removed_reason": "weaker_variant_same_region",
                "kept_region_winner": seen_regions[region_id].get("title"),
            })
            continue
        seen_regions[region_id] = clip
        clip["variant_winner"] = True
        region_winners.append(clip)

    vectors = _semantic_vectors(region_winners)
    semantic_kept = []
    semantic_vectors = []
    semantic_groups = []
    for clip, vector in zip(region_winners, vectors):
        similarities = [
            max(
                _cosine_similarity(vector, kept_vector),
                _token_containment_similarity(clip.get("text", ""), kept.get("text", "")),
            )
            for kept, kept_vector in zip(semantic_kept, semantic_vectors)
        ]
        best_similarity = max(similarities, default=0.0)
        if best_similarity >= 0.72:
            kept_index = similarities.index(best_similarity)
            removed.append({
                "candidate": clip,
                "removed_reason": "semantic_duplicate",
                "similarity": round(best_similarity, 4),
                "kept_candidate": semantic_kept[kept_index].get("title"),
            })
            semantic_groups.append({
                "kept": semantic_kept[kept_index].get("title"),
                "removed": clip.get("title"),
                "similarity": round(best_similarity, 4),
            })
            continue
        semantic_kept.append(clip)
        semantic_vectors.append(vector)

    final = []
    remaining = list(zip(semantic_kept, semantic_vectors))
    diversity_decisions = []
    while remaining and len(final) < 5:
        scored = []
        for clip, vector in remaining:
            similarities = [
                _cosine_similarity(vector, selected_vector)
                for _, selected_vector in final
            ]
            similarity_penalty = max(similarities, default=0.0) * 4.0
            same_type_penalty = 2.0 if any(
                selected.get("clip_type") == clip.get("clip_type")
                for selected, _ in final
            ) else 0.0
            effective = float(clip.get("score", 0)) - similarity_penalty - same_type_penalty
            scored.append((effective, clip, vector, similarity_penalty, same_type_penalty))
        effective, chosen, chosen_vector, similarity_penalty, type_penalty = max(
            scored, key=lambda item: (item[0], float(item[1].get("score", 0)))
        )
        final.append((chosen, chosen_vector))
        remaining = [item for item in remaining if item[0] is not chosen]
        diversity_decisions.append({
            "candidate": chosen.get("title"),
            "original_score": int(chosen.get("score", 0)),
            "effective_score": round(effective, 3),
            "semantic_penalty": round(similarity_penalty, 3),
            "same_type_penalty": round(type_penalty, 3),
        })

    final_clips = [clip for clip, _ in final]
    displayed_vectors = [vector for _, vector in final]
    displayed_regions = {
        clip.get("region_id") for clip in final_clips if clip.get("region_id") is not None
    }
    review_candidates.sort(key=lambda clip: (
        clip.get("decision") != "BORDERLINE",
        -int(clip.get("score", 0)),
        -float(clip.get("confidence_score", 0)),
    ))
    review_vectors = _semantic_vectors(review_candidates)
    review_pool = list(zip(review_candidates, review_vectors))

    # Prefer distinct regions first, then allow the best remaining review item.
    for prefer_new_region in (True, False):
        for clip, vector in review_pool:
            if len(final_clips) >= 5:
                break
            if any(existing.get("candidate_id") == clip.get("candidate_id") for existing in final_clips):
                continue
            region_id = clip.get("region_id")
            if prefer_new_region and region_id is not None and region_id in displayed_regions:
                continue
            similarity = max(
                (
                    max(
                        _cosine_similarity(vector, displayed_vector),
                        _token_containment_similarity(
                            clip.get("text", ""), displayed.get("text", "")
                        ),
                    )
                    for displayed, displayed_vector in zip(final_clips, displayed_vectors)
                ),
                default=0.0,
            )
            if similarity >= 0.82:
                continue
            experimental = dict(clip)
            experimental["quality_tier"] = "experimental"
            experimental["dashboard_experimental"] = True
            experimental["review_original_decision"] = clip.get("decision")
            experimental["review_similarity"] = round(similarity, 4)
            final_clips.append(experimental)
            displayed_vectors.append(vector)
            if region_id is not None:
                displayed_regions.add(region_id)

    for display_rank, clip in enumerate(final_clips, start=1):
        clip["display_rank"] = display_rank
    final_ids = {clip.get("candidate_id") for clip in final_clips}
    for clip in semantic_kept:
        if clip.get("candidate_id") not in final_ids:
            removed.append({
                "candidate": clip,
                "removed_reason": "low_score" if int(clip.get("score", 0)) < 65 else "diversity_rerank",
            })
    visible_ids = {clip.get("candidate_id") for clip in final_clips}
    candidate_decisions = []
    for clip in all_candidates:
        record = candidate_ranking_record(clip)
        record["dashboard_visible"] = clip.get("candidate_id") in visible_ids
        record["dashboard_experimental"] = bool(
            clip.get("candidate_id") in visible_ids and clip.get("decision") != "KEEP"
        )
        candidate_decisions.append(record)
    debug = {
        "region_comparison": [
            {
                "region_id": region_id,
                "variant_winner": winner.get("variant_type"),
                "winner_score": int(winner.get("score", 0)),
            }
            for region_id, winner in seen_regions.items()
        ],
        "semantic_duplicate_groups": semantic_groups,
        "diversity_decisions": diversity_decisions,
        "removed_candidates": [
            {
                "title": item["candidate"].get("title"),
                "start": item["candidate"].get("start"),
                "end": item["candidate"].get("end"),
                "score": item["candidate"].get("score"),
                "region_id": item["candidate"].get("region_id"),
                "variant_type": item["candidate"].get("variant_type"),
                "decision": item["candidate"].get("decision", "KEEP"),
                "reject_reason": item["candidate"].get("reject_reason", ""),
                "relative_rank": item["candidate"].get("relative_rank"),
                "natural_hook_quality": item["candidate"].get("natural_hook_quality", 0),
                "hook_improved_quality": item["candidate"].get("hook_improved_quality", 0),
                "removed_reason": item["removed_reason"],
                **({"similarity": item["similarity"]} if "similarity" in item else {}),
            }
            for item in removed
        ],
        "candidate_decisions": candidate_decisions,
        "ai_candidates_received": len(all_candidates),
        "ai_keep_count": sum(clip.get("decision") == "KEEP" for clip in all_candidates),
        "ai_borderline_count": sum(clip.get("decision") == "BORDERLINE" for clip in all_candidates),
        "ai_reject_count": sum(clip.get("decision") == "REJECT" for clip in all_candidates),
        "publishable_displayed_count": sum(
            clip.get("decision") == "KEEP" for clip in final_clips
        ),
        "experimental_displayed_count": sum(
            bool(clip.get("dashboard_experimental")) for clip in final_clips
        ),
        "final_displayed_count": len(final_clips),
    }
    print(f"FINAL DISPLAYED CLIPS: {len(final_clips)}", flush=True)
    return final_clips, debug


def openai_error_details(error):
    body = getattr(error, "body", None)
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)

    if isinstance(body, dict):
        nested_error = body.get("error")
        nested_error = nested_error if isinstance(nested_error, dict) else {}
        code = body.get("code") or nested_error.get("code")
        message = body.get("message") or nested_error.get("message")
    else:
        code = getattr(error, "code", None)
        message = None
    message = message or getattr(error, "message", None) or str(error)

    # Defensive redaction in case an upstream exception ever includes credentials.
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        message = str(message).replace(api_key, "[REDACTED]")

    return status, code, str(message)


def print_openai_error(error, reason=None):
    status, code, message = openai_error_details(error)
    fallback_reason = reason or f"{type(error).__name__}: {message}"
    print(f"AI SCORING FALLBACK: {fallback_reason}", file=sys.stderr, flush=True)
    print(f"EXCEPTION TYPE: {type(error).__name__}", file=sys.stderr, flush=True)
    print(f"HTTP STATUS: {status if status is not None else 'unknown'}", file=sys.stderr, flush=True)
    print(f"ERROR CODE: {code if code is not None else 'unknown'}", file=sys.stderr, flush=True)
    print(f"ERROR MESSAGE: {message}", file=sys.stderr, flush=True)


def _response_incomplete_reason(response):
    status = getattr(response, "status", None)
    if status in (None, "completed"):
        return None
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    if reason is None and isinstance(details, dict):
        reason = details.get("reason")
    return f"response status {status}" + (f" ({reason})" if reason else "")


def _log_ai_response(response):
    status = getattr(response, "status", None)
    normalized_status = "complete" if status == "completed" else (status or "unknown")
    print(f"AI OUTPUT STATUS: {normalized_status}", flush=True)
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None and isinstance(usage, dict):
        output_tokens = usage.get("output_tokens")
    if output_tokens is not None:
        print(f"AI OUTPUT TOKENS: {output_tokens}", flush=True)


def _incomplete_detail_reason(response):
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    if reason is None and isinstance(details, dict):
        reason = details.get("reason")
    return reason


def _strict_json_from_response_text(response):
    incomplete_reason = _response_incomplete_reason(response)
    if incomplete_reason:
        raise ValueError(f"Incomplete structured response: {incomplete_reason}")
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Structured response contained no JSON text")
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(output_text.strip())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid or truncated structured JSON at character {error.pos}: {error.msg}"
        ) from error
    if output_text.strip()[end:].strip():
        raise ValueError("Structured response contained trailing non-JSON content")
    if not isinstance(parsed, dict):
        raise ValueError("Structured response root must be an object")
    return parsed


def _ai_scoring_models():
    from typing import Literal
    from pydantic import BaseModel, ConfigDict, Field

    class ClipEvaluation(BaseModel):
        model_config = ConfigDict(extra="forbid")

        candidate_id: str
        decision: Literal["KEEP", "REJECT"]
        title: str = Field(max_length=80)
        scroll_stop_score: int = Field(ge=0, le=25)
        retention_score: int = Field(ge=0, le=20)
        payoff_score: int = Field(ge=0, le=15)
        emotion_novelty_score: int = Field(ge=0, le=15)
        standalone_score: int = Field(ge=0, le=10)
        shareability_score: int = Field(ge=0, le=10)
        pacing_score: int = Field(ge=0, le=5)
        natural_hook_quality: int = Field(ge=0, le=10)
        hook_improved_quality: int = Field(ge=0, le=10)
        viral_reason: str = Field(max_length=150)
        weakness: str = Field(max_length=100)
        reject_reason: str = Field(max_length=140)
        clip_type: Literal[
            "Big Number", "Wild Story", "Hot Take", "Conflict", "Reaction",
            "Funny", "Reveal", "Emotional", "Educational", "Prediction",
            "Founder Story", "Debate", "Unexpected", "Other",
        ]
        selected_hook_option_id: int | None = Field(ge=1, le=5)
        relative_rank: int = Field(ge=1)
        campaign_fit_score: int = Field(ge=0, le=10)

    class ClipEvaluationBatch(BaseModel):
        model_config = ConfigDict(extra="forbid")
        evaluations: list[ClipEvaluation]

    return ClipEvaluationBatch


def build_ai_scoring_instructions(ranking_options=None):
    options = normalize_ranking_options(ranking_options)
    if options["viral_shorts"]:
        objective = (
            "PRIMARY OBJECTIVE: Select clips optimized for short-form virality, "
            "not general informational quality."
        )
    else:
        objective = (
            "PRIMARY OBJECTIVE: Select only genuinely publishable standalone short-form "
            "moments; do not favor virality over useful information."
        )
    if options["two_second_hook"]:
        hook_rule = (
            "AUTO-HOOK OPTION IS ENABLED. A synthetic cold-open hook can be used, but only "
            "to improve an already strong segment. Distinguish natural_hook_quality from "
            "hook_improved_quality. Never turn a boring or payoff-free segment into KEEP merely "
            "because one interesting sentence can be moved to the front. Select only one supplied "
            "hook_option id from the same candidate, or null."
        )
    else:
        hook_rule = (
            "AUTO-HOOK OPTION IS DISABLED. selected_hook_option_id must be null and "
            "hook_improved_quality must equal natural_hook_quality."
        )

    return " ".join((
        "Act as a strict human short-form editor making publish-or-discard decisions, not as a "
        "reviewer trying to find something positive in every excerpt.",
        objective,
        global_best_moments_prompt_block(),
        "Evaluate every supplied candidate in this one batch. Compare the candidates against each "
        "other before assigning final scores. First identify internally the strongest moments, "
        "which candidates tell substantially the same story, which are only acceptable in isolation "
        "but clearly weaker in this set, and which a human editor would actually publish. Keep the "
        "absolute 0-100 standard, but calibrate scores and relative_rank against this video's complete "
        "candidate set. There is no quota: many or all candidates may be REJECT, and zero or one KEEP "
        "is valid. Never keep a candidate just to fill a top five. KEEP means BEST PUBLISHABLE among "
        "moments that clear the publish bar. A REJECT may still receive honest non-zero component "
        "scores when it is USEFUL FOR HUMAN REVIEW; do not collapse all rejected candidates to near "
        "zero merely because they were rejected.",
        "For every candidate decide decision KEEP or REJECT first. KEEP only when: (1) its first 1-2 "
        "seconds contain an understandable scroll-stop, or a credible supplied auto-hook improves an "
        "already strong segment; (2) it creates a concrete reason to continue through curiosity, story, "
        "conflict, surprise, a strong claim, unusual visuals/events, or real stakes; (3) it reaches a "
        "clear payoff such as a reveal, answer, reaction, punchline, result, or strong concluding claim; "
        "(4) it works mostly without knowledge of the source video; and (5) it contains enough genuine "
        "content for a short AND is among the best moments relative to the full candidate set.",
        "REJECT candidates that are mainly setup, introductions, transitions, smalltalk, descriptions "
        "without payoff, generic information, weak reactions, context-dependent fragments, clips whose "
        "interesting event happens after the end, clips whose first interesting sentence arrives too "
        "late, or clips an auto-hook would merely disguise. Greetings, channel/video intros, sponsor "
        "language, self-promotion, slow setup, repetition, dead air, mid-thought starts, missing context, "
        "and endings before payoff are hard negatives. Map ordinary intro/setup material to Other. An "
        "exceptional early reveal, conflict, or payoff may still qualify.",
        "Apply this viral-short test internally to each candidate: A) Without a title, would seconds 1-2 "
        "stop a stranger? B) What exactly does the viewer want to learn next? If there is no concrete "
        "answer, reduce retention sharply. C) What reward does the viewer receive at the end? If none, "
        "reduce payoff sharply. D) What single sentence or moment would prompt a share or comment? If "
        "none, reduce shareability sharply. E) Is this genuinely among the best moments in the complete "
        "set? If not, do not inflate it.",
        "Use transcript, region_id and variant context, dead-air ratio, audio, visual, duration, "
        "surrounding context, and source position. Compare SHORT, STANDARD, and EXTENDED variants in the "
        "same region; prefer the version that reaches the payoff with least unnecessary setup. Longer is "
        "not automatically better. Audio and visual signals may raise or lower potential, but noisy motion "
        "alone is not meaningful and emotions must not be invented.",
        hook_rule,
        calibrated_scoring_bands_prompt(),
        "Every candidate receives a positive unique relative_rank from 1 to N (1 is best across the full "
        "set, including REJECT). REJECT still needs a concrete reject_reason; KEEP must use an empty "
        "reject_reason. Titles must be natural, specific, in the spoken language, and at most 8 words. "
        "viral_reason is at most 18 words and weakness at most 12 words. Never repeat the transcript or "
        "include extended quotations.",
        "If campaign context exists, score campaign_fit_score 0-10 separately. Campaign relevance cannot "
        "raise the 100-point viral score or rescue a bad clip. Without campaign context, return 0. Return "
        "every candidate exactly once in the strict schema.",
    ))



def rerank_candidates_with_openai(candidates, campaign=None, ranking_options=None):
    if not candidates:
        return candidates, False
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "AI SCORING FALLBACK: OPENAI_API_KEY is not configured",
            file=sys.stderr,
            flush=True,
        )
        return candidates, False

    try:
        from openai import OpenAI

        ranking_options = normalize_ranking_options(ranking_options)
        payload = ai_candidate_payload(candidates, campaign, ranking_options)
        client = OpenAI()
        instructions = build_ai_scoring_instructions(ranking_options)
        serialized_payload = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        print(f"AI INPUT CANDIDATES: {len(candidates)}", flush=True)
        print(f"AI CANDIDATES RECEIVED: {len(candidates)}", flush=True)
        scoring_model = _ai_scoring_models()
        for attempt, token_budget in enumerate((12000, 20000), start=1):
            try:
                print(f"AI MAX OUTPUT TOKENS: {token_budget}", flush=True)
                response = client.responses.create(
                    model="gpt-5-mini",
                    store=False,
                    instructions=instructions,
                    input=serialized_payload,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "clip_candidate_rankings",
                            "strict": True,
                            "schema": ai_scoring_schema(len(candidates)),
                        }
                    },
                    max_output_tokens=token_budget,
                )
                _log_ai_response(response)
                incomplete_reason = _response_incomplete_reason(response)
                if incomplete_reason:
                    detail_reason = _incomplete_detail_reason(response)
                    if detail_reason == "max_output_tokens":
                        print(
                            "AI RESPONSE INCOMPLETE: max_output_tokens",
                            file=sys.stderr,
                            flush=True,
                        )
                    raise ValueError(f"Incomplete structured response: {incomplete_reason}")
                output_text = getattr(response, "output_text", None)
                if not isinstance(output_text, str) or not output_text.strip():
                    raise ValueError("Structured response contained no JSON output")
                parsed_model = scoring_model.model_validate_json(output_text)
                parsed = parsed_model.model_dump()
                ranked = apply_ai_evaluations(
                    candidates,
                    parsed["evaluations"],
                    allow_synthetic_hook=ranking_options["two_second_hook"],
                )
                keep_count = sum(clip.get("decision") == "KEEP" for clip in ranked)
                borderline_count = sum(
                    clip.get("decision") == "BORDERLINE" for clip in ranked
                )
                reject_count = len(ranked) - keep_count - borderline_count
                print(f"AI KEEP: {keep_count}", flush=True)
                print(f"AI BORDERLINE: {borderline_count}", flush=True)
                print(f"AI REJECT: {reject_count}", flush=True)
                print(
                    f"AI SCORING ACTIVE: gpt-5-mini ranked {len(ranked)} candidates",
                    flush=True,
                )
                return ranked, True
            except Exception as error:
                if attempt == 1:
                    _, _, message = openai_error_details(error)
                    print(
                        "AI SCORING RETRY: bundled request failed once: "
                        f"{type(error).__name__}: {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                raise
    except Exception as error:
        print_openai_error(error, f"bundled structured request failed: {error}")
        return candidates, False


def surrounding_context_summary(segments, start, end):
    before = [
        str(segment.get("text", "")).strip() for segment in segments
        if float(segment.get("end", 0)) <= start
    ][-2:]
    after = [
        str(segment.get("text", "")).strip() for segment in segments
        if float(segment.get("start", 0)) >= end
    ][:2]
    return {
        "before": " ".join(before)[-240:],
        "after": " ".join(after)[:240],
    }


def load_multimodal_candidates(project_dir, segments):
    """Load the locally merged source mix without additional media or API work."""
    features_path = project_dir / "analysis_features.json"
    try:
        features = json.loads(features_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return build_clip_candidates(segments)[:15]

    from semantic_candidates import augment_project
    features = augment_project(project_dir)

    selected = features.get("selected_candidates")
    source_duration = float(features.get("duration") or 0)
    if not isinstance(selected, list) or not selected:
        return build_clip_candidates(segments)[:15]

    candidates = []
    for item in selected[:20]:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            text = str(item["text"]).strip()
            local_score = int(item.get("local_multimodal_score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if not text or end <= start or not 15 <= end - start <= 65.5:
            continue
        candidates.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "text": text,
            "score": max(0, min(100, local_score)),
            "score_breakdown": {
                "multimodal_local": max(0, min(100, local_score)),
                "seed_modalities": item.get("seed_modalities", []),
            },
            "audio": item.get("audio", {}),
            "visual": item.get("visual", {}),
            "source_position": start / source_duration if source_duration > 0 else 0.0,
            "candidate_source": item.get("candidate_source", "multimodal"),
            "semantic_candidate_id": item.get("semantic_candidate_id"),
            "semantic_interest_score": item.get("semantic_interest_score"),
            "detected_signals": item.get("detected_signals", []),
            "central_claim": item.get("central_claim"),
            "region_id": item.get("region_id"),
            "region_rank": item.get("region_rank", item.get("region_id")),
            "region_score": float(item.get("local_interest_score", 0) or 0),
            "local_interest_score": float(item.get("local_interest_score", 0) or 0),
            "variant_type": item.get("region_type", "STANDARD"),
            "dead_air_ratio": float(item.get("dead_air_ratio", 0) or 0),
            "anchor_types": item.get("anchor_types", item.get("seed_modalities", [])),
            "anchor_strengths": item.get("anchor_strengths", {}),
            "surrounding_context": surrounding_context_summary(segments, start, end),
        })
    if not candidates:
        return build_clip_candidates(segments)[:15]

    # Claim-aware boundary resolution only — does not change ranking/selection logic.
    transcript = load_transcript(project_dir) or {}
    boundary_context = build_boundary_context(project_dir, transcript)
    if not boundary_context.get("segments"):
        boundary_context["segments"] = [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": str(s.get("text", "")).strip(),
            }
            for s in segments
            if float(s.get("end", 0) or 0) > float(s.get("start", 0) or 0)
        ]
    return apply_claim_aware_boundaries(candidates, boundary_context, dedupe=True) or candidates


def candidate_debug_frame_times(candidate, analysis_features):
    start = float(candidate["start"])
    end = float(candidate["end"])
    peak_candidates = []
    for event in analysis_features.get("audio_events", []):
        event_time = float(event.get("time", -1) or -1)
        if start <= event_time <= end:
            strength = max(
                float(event.get("peak_energy", 0) or 0),
                float(event.get("audio_energy", 0) or 0),
                float(event.get("mean_energy", 0) or 0),
            )
            strength += 0.20 if event.get("type") == "energy_spike" else 0.0
            strength += 0.12 if event.get("rapid_speech") else 0.0
            peak_candidates.append((strength, event_time, "audio"))
    for event in analysis_features.get("visual_events", []):
        event_time = float(event.get("time", -1) or -1)
        if start <= event_time <= end:
            strength = (
                float(event.get("visual_change", 0) or 0) * 0.60
                + float(event.get("motion_score", 0) or 0) * 0.25
                + (0.25 if event.get("scene_change") else 0.0)
                + (0.10 if event.get("face_count_change") else 0.0)
            )
            peak_candidates.append((strength, event_time, "visual"))
    if peak_candidates:
        _, peak_time, peak_source = max(peak_candidates, key=lambda item: item[0])
    else:
        seed_time = float(candidate.get("seed_time", (start + end) / 2) or 0)
        peak_time = seed_time if start <= seed_time <= end else (start + end) / 2
        peak_source = "seed_or_midpoint"
    return {
        "start": round(min(end, start + 0.35), 3),
        "peak": round(max(start, min(end, peak_time)), 3),
        "end": round(max(start, end - 0.35), 3),
        "peak_source": peak_source,
    }


def extract_candidate_debug_frames(project_dir, candidates, analysis_features, enabled=True):
    result = {"enabled": bool(enabled), "candidates": {}}
    if not enabled:
        return result
    ffmpeg = shutil.which("ffmpeg")
    source_candidates = sorted(project_dir.glob("source.*"))
    if not ffmpeg or not source_candidates:
        result["error"] = (
            "ffmpeg not found" if not ffmpeg else "project source video not found"
        )
        print(f"CANDIDATE DEBUG FRAMES DISABLED: {result['error']}", flush=True)
        return result

    source_video = source_candidates[0]
    frames_root = project_dir / "analysis_frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = f"candidate_{index}"
        output_dir = frames_root / candidate_id
        output_dir.mkdir(parents=True, exist_ok=True)
        times = candidate_debug_frame_times(candidate, analysis_features)
        candidate_result = {
            "times": times,
            "files": {},
            "errors": [],
        }
        for label in ("start", "peak", "end"):
            target = output_dir / f"{label}.jpg"
            relative_target = target.relative_to(project_dir).as_posix()
            if target.is_file() and target.stat().st_size > 0:
                candidate_result["files"][label] = relative_target
                continue
            temporary = output_dir / f"{label}.tmp.jpg"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-ss", f"{times[label]:.3f}", "-i", str(source_video),
                "-frames:v", "1", "-vf",
                "scale=512:-2:force_original_aspect_ratio=decrease",
                "-q:v", "4", str(temporary),
            ]
            try:
                completed = subprocess.run(command, capture_output=True, text=True)
            except OSError as frame_error:
                candidate_result["errors"].append(
                    f"{label}: {type(frame_error).__name__}: {frame_error}"
                )
                continue
            if completed.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                temporary.replace(target)
                candidate_result["files"][label] = relative_target
            else:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                detail = (completed.stderr or completed.stdout or "frame extraction failed").strip()
                candidate_result["errors"].append(f"{label}: {detail[:240]}")
        result["candidates"][candidate_id] = candidate_result
    generated = sum(
        len(item.get("files", {})) for item in result["candidates"].values()
    )
    result["generated_or_reused_frames"] = generated
    print(f"CANDIDATE DEBUG FRAMES: {generated}", flush=True)
    return result


def save_ai_candidate_debug(
    project_dir, candidates, campaign, ranking_options, selection_debug,
    ai_active, frame_debug,
):
    decisions = {
        item.get("candidate_id"): item
        for item in (selection_debug or {}).get("candidate_decisions", [])
    }
    payload = ai_candidate_payload(candidates, campaign, ranking_options)
    debug_candidates = []
    for index, item in enumerate(payload["candidates"], start=1):
        candidate_id = item["candidate_id"]
        decision = decisions.get(candidate_id, {})
        source_candidate = candidates[index - 1]
        debug_candidates.append({
            "candidate_id": candidate_id,
            "candidate_source": source_candidate.get("candidate_source", "multimodal"),
            "semantic_interest_score": source_candidate.get("semantic_interest_score"),
            "central_claim": source_candidate.get("central_claim"),
            "detected_signals": source_candidate.get("detected_signals", []),
            "transcript": item["transcript"],
            "start": item["start"],
            "end": item["end"],
            "duration": item["duration"],
            "region_id": item["region"].get("region_id"),
            "variant_type": item["region"].get("variant_type"),
            "local_interest_score": item["region"].get("local_interest_score"),
            "audio_summary": item["audio"],
            "visual_summary": item["visual"],
            "surrounding_context": item["surrounding_context"],
            "hook_options_sent_to_gpt": item["hook_options"],
            "hook_options_available": [
                {
                    "id": option_index,
                    "text": option.get("text", ""),
                    "start": option.get("start"),
                    "end": option.get("end"),
                    "local_strength": option.get("score"),
                }
                for option_index, option in enumerate(
                    source_candidate.get("_hook_options", []), start=1
                )
            ],
            "gpt_component_scores": decision.get("gpt_component_scores", {}),
            "post_penalty_component_scores": decision.get("component_scores", {}),
            "local_penalties": decision.get("penalties", []),
            "quality_gate_failures": decision.get("quality_gate_failures", []),
            "borderline_signals": decision.get("borderline_signals", []),
            "raw_score": decision.get("raw_score"),
            "score_cap": decision.get("score_cap"),
            "score_cap_reasons": decision.get("score_cap_reasons", []),
            "final_score": decision.get("score"),
            "confidence_score": decision.get("confidence_score"),
            "confidence_breakdown": decision.get("confidence_breakdown", {}),
            "evidence": decision.get("evidence", {}),
            "gpt_decision": decision.get("model_decision"),
            "decision": decision.get("decision"),
            "reject_reason": decision.get("reject_reason", ""),
            "relative_rank": decision.get("relative_rank"),
            "dashboard_visible": bool(decision.get("dashboard_visible")),
            "dashboard_experimental": bool(decision.get("dashboard_experimental")),
            "debug_frames": (frame_debug or {}).get("candidates", {}).get(candidate_id, {}),
        })
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ai_ranking_version": AI_RANKING_VERSION,
        "ai_scoring_active": bool(ai_active),
        "ranking_options": normalize_ranking_options(ranking_options),
        "candidate_count": len(debug_candidates),
        "counts": {
            "keep": (selection_debug or {}).get("ai_keep_count", 0),
            "borderline": (selection_debug or {}).get("ai_borderline_count", 0),
            "reject": (selection_debug or {}).get("ai_reject_count", 0),
            "dashboard_visible": (selection_debug or {}).get("final_displayed_count", 0),
            "dashboard_experimental": (selection_debug or {}).get(
                "experimental_displayed_count", 0
            ),
        },
        "visual_semantics_note": (
            "Current visual summaries describe cuts, motion, zoom-like changes and face-count "
            "changes; they do not identify rooms, objects, reveals or human reactions."
        ),
        "frame_extraction": frame_debug,
        "candidates": debug_candidates,
    }
    target = project_dir / "ai_candidate_debug.json"
    temporary = project_dir / "ai_candidate_debug.json.tmp"
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def save_final_analysis_ranking(project_dir, ranked, ai_active, selection_debug=None):
    """Append final ranking data to the per-project debug artifact atomically."""
    path = project_dir / "analysis_features.json"
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("analysis_features.json root is not an object")
        else:
            now = datetime.now(timezone.utc).isoformat()
            data = {
                "analysis_version": 2,
                "success": False,
                "error": "analysis_features.json was missing before final ranking update",
                "started_at": now,
                "completed_at": now,
                "diagnostics": {},
            }
        ranking_candidates = [
            candidate_ranking_record(clip, display_rank=index)
            for index, clip in enumerate(ranked, start=1)
        ]
        all_ai_candidates = (selection_debug or {}).get(
            "candidate_decisions", ranking_candidates
        )
        data["final_ranking"] = {
            "mode": "gpt-5-mini" if ai_active else "local",
            "candidates": ranking_candidates,
        }
        data["ai_ranking"] = {
            "analysis_pipeline_version": ANALYSIS_PIPELINE_VERSION,
            "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
            "ai_ranking_version": AI_RANKING_VERSION,
            "mode": "gpt-5-mini" if ai_active else "local",
            "ranking_options": normalize_ranking_options(
                read_json_safely(project_dir / "ranking_options.json", {})
            ),
            "candidate_scores": all_ai_candidates,
            "region_comparison": (selection_debug or {}).get("region_comparison", []),
            "semantic_duplicate_groups": (selection_debug or {}).get("semantic_duplicate_groups", []),
            "diversity_decisions": (selection_debug or {}).get("diversity_decisions", []),
            "removed_candidates": (selection_debug or {}).get("removed_candidates", []),
            "final_top_10": ranking_candidates[:10],
            "candidates_received": (selection_debug or {}).get(
                "ai_candidates_received", len(all_ai_candidates)
            ),
            "keep_count": (selection_debug or {}).get("ai_keep_count", len(ranked)),
            "borderline_count": (selection_debug or {}).get("ai_borderline_count", 0),
            "reject_count": (selection_debug or {}).get("ai_reject_count", 0),
            "experimental_displayed_count": (selection_debug or {}).get(
                "experimental_displayed_count", 0
            ),
            "final_displayed_count": len(ranking_candidates),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

        tier_counts = {
            tier: sum(clip.get("quality_tier") == tier for clip in ranked)
            for tier in ("strong", "good", "experimental", "weak")
        }
        summary = {
            "analysis_pipeline_version": ANALYSIS_PIPELINE_VERSION,
            "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
            "ai_ranking_version": AI_RANKING_VERSION,
            "top_candidate": ranking_candidates[0] if ranking_candidates else None,
            "strong_candidates": tier_counts["strong"],
            "good_candidates": tier_counts["good"],
            "experimental_candidates": tier_counts["experimental"],
            "weak_candidates": tier_counts["weak"],
            "dominant_clip_types": list(dict.fromkeys(
                clip.get("clip_type") for clip in ranked if clip.get("clip_type")
            ))[:5],
            "regions_considered": len(data.get("regions_selected_for_visual_analysis", [])),
            "variants_considered": len(data.get("raw_candidate_variants", [])),
            "candidates_sent_to_ai": len(data.get("candidates_sent_to_ai", [])),
            "ai_keep": (selection_debug or {}).get("ai_keep_count", len(ranked)),
            "ai_borderline": (selection_debug or {}).get("ai_borderline_count", 0),
            "ai_reject": (selection_debug or {}).get("ai_reject_count", 0),
            "experimental_displayed_clips": (selection_debug or {}).get(
                "experimental_displayed_count", 0
            ),
            "final_displayed_clips": len(ranking_candidates),
        }
        summary_path = project_dir / "selection_summary.json"
        summary_temporary = project_dir / "selection_summary.json.tmp"
        summary_temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_temporary.replace(summary_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"WARNING: Could not update analysis debug ranking: {error}", flush=True)


def ensure_failed_analysis_features(project_dir, error, started_at, elapsed_seconds):
    """Create the required debug artifact if multimodal analysis could not do so."""
    path = project_dir / "analysis_features.json"
    if path.is_file():
        return
    payload = {
        "analysis_version": 2,
        "success": False,
        "error": f"{type(error).__name__}: {error}",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics": {
            "total_elapsed_seconds": round(float(elapsed_seconds), 3),
        },
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def ranked_clip_candidates(project_dir, segments, words):
    candidates = load_multimodal_candidates(project_dir, segments)
    campaign = read_json_safely(project_dir / "campaign.json", {})
    analysis_features = read_json_safely(project_dir / "analysis_features.json", {})
    analysis_features = analysis_features if isinstance(analysis_features, dict) else {}
    ranking_options = normalize_ranking_options(
        read_json_safely(project_dir / "ranking_options.json", {})
    )
    for candidate in candidates:
        candidate["_hook_options"] = build_hook_options(candidate, words)
    payload = ai_candidate_payload(candidates, campaign, ranking_options)
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    frame_debug = extract_candidate_debug_frames(
        project_dir,
        candidates,
        analysis_features,
        enabled=ranking_options["candidate_debug_frames"],
    )
    cache_key = (
        f"ai_ranking_{ANALYSIS_PIPELINE_VERSION}_{CANDIDATE_PIPELINE_VERSION}_"
        f"{AI_RANKING_VERSION}_{AI_CARD_OUTPUT_VERSION}_"
        f"{project_dir.name}_{fingerprint}"
    )
    cached = st.session_state.get(cache_key)
    if cached:
        cached_debug = cached.get("selection_debug", {})
        print(
            f"AI CANDIDATES RECEIVED: {cached_debug.get('ai_candidates_received', len(candidates))}",
            flush=True,
        )
        print(f"AI KEEP: {cached_debug.get('ai_keep_count', len(cached['clips']))}", flush=True)
        print(f"AI BORDERLINE: {cached_debug.get('ai_borderline_count', 0)}", flush=True)
        print(f"AI REJECT: {cached_debug.get('ai_reject_count', 0)}", flush=True)
        print(f"FINAL DISPLAYED CLIPS: {len(cached['clips'])}", flush=True)
        save_ai_candidate_debug(
            project_dir, candidates, campaign, ranking_options,
            cached_debug, cached["ai_active"], frame_debug,
        )
        return cached["clips"], cached["ai_active"]
    persistent_cache_path = project_dir / "ai_ranking_cache.json"
    persistent_cache = read_json_safely(persistent_cache_path, {})
    if (
        isinstance(persistent_cache, dict)
        and persistent_cache.get("analysis_pipeline_version") == ANALYSIS_PIPELINE_VERSION
        and persistent_cache.get("candidate_pipeline_version") == CANDIDATE_PIPELINE_VERSION
        and persistent_cache.get("ai_ranking_version") == AI_RANKING_VERSION
        and persistent_cache.get("fingerprint") == fingerprint
        and isinstance(persistent_cache.get("clips"), list)
    ):
        cached_clips = persistent_cache["clips"]
        cached_debug = persistent_cache.get("selection_debug", {})
        st.session_state[cache_key] = {
            "clips": cached_clips,
            "ai_active": True,
            "selection_debug": cached_debug,
        }
        print("AI RANKING CACHE HIT", flush=True)
        print(
            f"AI CANDIDATES RECEIVED: {cached_debug.get('ai_candidates_received', len(candidates))}",
            flush=True,
        )
        print(f"AI KEEP: {cached_debug.get('ai_keep_count', len(cached_clips))}", flush=True)
        print(f"AI BORDERLINE: {cached_debug.get('ai_borderline_count', 0)}", flush=True)
        print(f"AI REJECT: {cached_debug.get('ai_reject_count', 0)}", flush=True)
        print(f"FINAL DISPLAYED CLIPS: {len(cached_clips)}", flush=True)
        save_ai_candidate_debug(
            project_dir, candidates, campaign, ranking_options,
            cached_debug, True, frame_debug,
        )
        return cached_clips, True

    ai_started = time.perf_counter()
    # Keep the historical request record intact during an offline semantic audit.
    # Only the existing ranking workflow advances it to the current selection.
    analysis_features["candidates_sent_to_ai"] = analysis_features.get("selected_candidates", [])
    analysis_features["candidate_selection_status"] = "ranking_requested"
    features_target = project_dir / "analysis_features.json"
    features_temporary = features_target.with_suffix(".json.tmp")
    features_temporary.write_text(json.dumps(analysis_features, ensure_ascii=False, indent=2), encoding="utf-8")
    features_temporary.replace(features_target)
    ranked, ai_active = rerank_candidates_with_openai(
        candidates, campaign, ranking_options
    )
    ranked = apply_local_hook_defaults(
        ranked, enabled=ranking_options["two_second_hook"]
    )
    ranked, selection_debug = postprocess_ranked_candidates(ranked, campaign)
    ai_elapsed = time.perf_counter() - ai_started
    generation_state = st.session_state.get("_generation_timing")
    if (
        isinstance(generation_state, dict)
        and generation_state.get("project_id") == project_dir.name
    ):
        generation_state.setdefault("breakdown", {})["ai_scoring_seconds"] = ai_elapsed
    save_final_analysis_ranking(
        project_dir, ranked, ai_active, selection_debug=selection_debug
    )
    save_ai_candidate_debug(
        project_dir, candidates, campaign, ranking_options,
        selection_debug, ai_active, frame_debug,
    )
    if ai_active:
        st.session_state[cache_key] = {
            "clips": ranked,
            "ai_active": True,
            "selection_debug": selection_debug,
        }
        try:
            persistent_temporary = project_dir / "ai_ranking_cache.json.tmp"
            persistent_temporary.write_text(
                json.dumps({
                    "analysis_pipeline_version": ANALYSIS_PIPELINE_VERSION,
                    "candidate_pipeline_version": CANDIDATE_PIPELINE_VERSION,
                    "ai_ranking_version": AI_RANKING_VERSION,
                    "fingerprint": fingerprint,
                    "clips": ranked,
                    "selection_debug": selection_debug,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            persistent_temporary.replace(persistent_cache_path)
        except (OSError, TypeError, ValueError) as cache_error:
            print(f"WARNING: Could not save AI ranking cache: {cache_error}", flush=True)
    return ranked, ai_active


def load_transcript(project_dir):
    if project_dir is None:
        return None

    transcript_path = project_dir / "transcript.json"

    if not transcript_path.exists():
        return None

    with transcript_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_segments(selected_clips, project_dir):
    data = []

    for index, clip in enumerate(selected_clips, start=1):
        data.append({
            "start": clip["start"],
            "end": clip["end"],
            "hook": clip["text"][:120].strip(),
            "overall": clip["score"],
            "title": f"Clip {index}",
            "hook_enabled": bool(clip.get("hook_enabled")),
            "hook_start": float(clip.get("hook_start") or 0),
            "hook_end": float(clip.get("hook_end") or 0),
            "hook_text": str(clip.get("hook_text") or ""),
            "hook_reason": str(clip.get("hook_reason") or ""),
        })

    with (project_dir / "segments.json").open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def hook_setting_id(clip):
    return f"{float(clip['start']):.3f}-{float(clip['end']):.3f}"


def load_hook_settings(project_dir):
    try:
        data = json.loads((project_dir / "hook_settings.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_hook_setting(project_dir, clip, enabled):
    settings = load_hook_settings(project_dir)
    setting_id = hook_setting_id(clip)
    value = {
        "hook_enabled": bool(enabled),
        "hook_start": float(clip.get("hook_start") or 0),
        "hook_end": float(clip.get("hook_end") or 0),
        "hook_text": str(clip.get("hook_text") or ""),
        "hook_reason": str(clip.get("hook_reason") or ""),
    }
    if settings.get(setting_id) == value:
        return
    settings[setting_id] = value
    target = project_dir / "hook_settings.json"
    temporary = project_dir / "hook_settings.json.tmp"
    temporary.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)


def rendered_paths_for_clip(clip, project_dir):
    hook = clip["text"][:120].strip().lower()
    stem = re.sub(r"[^\w\s-]", "", hook)
    stem = re.sub(r"\s+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    stem = (stem[:70].strip("-") or "clip")
    output_dir = project_dir / "clips"
    return output_dir / f"{stem}.mp4", output_dir / f"{stem}.png"


BOUNDARY_OPTIONS = (
    "Original",
    "Start -5s",
    "Start -3s",
    "Start +3s",
    "End +5s",
    "End +10s",
    "Expand to natural ending",
)

BOUNDARY_VARIANTS = {
    "Original": "original",
    "Start -5s": "start-minus-5s",
    "Start -3s": "start-minus-3s",
    "Start +3s": "start-plus-3s",
    "End +5s": "end-plus-5s",
    "End +10s": "end-plus-10s",
    "Expand to natural ending": "natural-ending",
}


def preview_path_for_clip(clip, project_dir, aspect, variant="original"):
    rendered_clip, _ = rendered_paths_for_clip(clip, project_dir)
    preview_format = aspect.replace(":", "x")
    return (
        project_dir / "clips" / "previews" / preview_format
        / f"{rendered_clip.stem}__{variant}.mp4"
    )


def build_boundary_context(project_dir, transcript_data):
    """Collect existing timing signals without rerunning any analysis."""
    transcript_segments = []
    for segment in transcript_data.get("segments", []):
        try:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
        except (TypeError, ValueError):
            continue
        if end > start:
            transcript_segments.append({
                "start": start,
                "end": end,
                "text": str(segment.get("text", "")).strip(),
            })

    transcript_words = []
    for word in transcript_data.get("words", []) or []:
        if not isinstance(word, dict):
            continue
        try:
            start = float(word.get("start", -1))
            end = float(word.get("end", -1))
        except (TypeError, ValueError):
            continue
        token = str(word.get("word", word.get("text", ""))).strip()
        if token and end > start >= 0:
            transcript_words.append({"word": token, "start": start, "end": end})

    try:
        duration = float(transcript_data.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0 and transcript_segments:
        duration = max(segment["end"] for segment in transcript_segments)

    scene_changes = []
    try:
        features = json.loads(
            (project_dir / "analysis_features.json").read_text(encoding="utf-8")
        )
        for event in features.get("visual_events", []):
            if event.get("scene_change"):
                event_time = float(event.get("time", -1))
                if event_time >= 0:
                    scene_changes.append(event_time)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    return {
        "duration": duration,
        "segments": transcript_segments,
        "words": transcript_words,
        "scene_changes": sorted(scene_changes),
    }


def find_natural_ending(original_end, boundary_context, clip=None):
    """Recompute semantic completion via the shared claim-aware boundary engine."""
    start = float((clip or {}).get("start", 0.0) or 0.0)
    claim = (clip or {}).get("central_claim")
    variant = (clip or {}).get("variant_type") or (clip or {}).get("region_type") or "STANDARD"
    return engine_find_natural_ending(
        start,
        float(original_end),
        boundary_context,
        central_claim=claim if isinstance(claim, dict) else None,
        variant_mode=str(variant),
    )


def apply_boundary_mode(clip, mode, boundary_context):
    adjusted = dict(clip)
    original_start = float(clip["start"])
    original_end = float(clip["end"])
    start, end = original_start, original_end

    if mode == "Start -5s":
        start -= 5.0
    elif mode == "Start -3s":
        start -= 3.0
    elif mode == "Start +3s":
        start += 3.0
    elif mode == "End +5s":
        end += 5.0
    elif mode == "End +10s":
        end += 10.0
    elif mode == "Expand to natural ending":
        # Same claim-aware engine: preserve start, recompute end (expand or shrink).
        resolved = resolve_boundaries(
            clip,
            words=boundary_context.get("words"),
            segments=boundary_context.get("segments"),
            scene_changes=boundary_context.get("scene_changes"),
            source_duration=boundary_context.get("duration"),
            preserve_start=True,
            end_only=True,
        )
        start = float(resolved["start"])
        end = float(resolved["end"])
        adjusted["boundary_debug"] = resolved.get("boundary_debug")
        if resolved.get("text"):
            adjusted["text"] = resolved["text"]

    duration = float(boundary_context.get("duration") or 0)
    start = max(0.0, start)
    if duration > 0:
        end = min(end, duration)
    if end <= start:
        start, end = original_start, original_end

    # Manual nudges still must never cut a word when word timings exist.
    if mode not in {"Expand to natural ending", "Original"}:
        words = boundary_context.get("words") or []
        start = align_start_to_words(start, words)
        end = align_end_to_words(end, words, source_duration=duration or None)
        if end <= start:
            start, end = original_start, original_end

    adjusted.update({
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "boundary_mode": mode,
        "boundary_variant": BOUNDARY_VARIANTS.get(mode, "original"),
    })
    return adjusted


def format_clip_time(seconds):
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def is_valid_mp4(path):
    try:
        return path.suffix.lower() == ".mp4" and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def format_runtime(seconds):
    total = max(0, int(round(float(seconds or 0))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def load_project_timings(project_dir):
    if project_dir is None:
        return {}
    try:
        data = json.loads((project_dir / "timings.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_project_timing(project_dir, process_name, values):
    """Merge one process timing atomically without removing other measurements."""
    try:
        timings = load_project_timings(project_dir)
        timings[process_name] = values
        target = project_dir / "timings.json"
        temporary = project_dir / "timings.json.tmp"
        temporary.write_text(
            json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
    except (OSError, TypeError, ValueError) as error:
        print(f"WARNING: Could not save {process_name} timing: {error}", flush=True)


def complete_generation_timing(project_dir, success):
    state = st.session_state.get("_generation_timing")
    if not isinstance(state, dict) or state.get("project_id") != project_dir.name:
        return None
    total_seconds = max(0.0, time.perf_counter() - state["started_perf"])
    values = {
        "total_seconds": round(total_seconds, 3),
        "total_generation_seconds": round(total_seconds, 3),
        "success": bool(success),
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    breakdown = state.get("breakdown")
    if isinstance(breakdown, dict):
        values.update({key: round(float(value), 3) for key, value in breakdown.items()})
    save_project_timing(project_dir, "generation", values)
    st.session_state.pop("_generation_timing", None)
    return values


def render_single_preview(clip, project_dir, aspect, variant):
    preview_dir = project_dir / "clips" / "previews" / aspect.replace(":", "x")
    preview_dir.mkdir(parents=True, exist_ok=True)
    segment_file = preview_dir / f"_preview_segment_{variant}.json"
    preview_path = preview_path_for_clip(clip, project_dir, aspect, variant)
    rendered_clip, _ = rendered_paths_for_clip(clip, project_dir)
    generated_path = preview_dir / rendered_clip.name
    try:
        preview_path.unlink()
    except FileNotFoundError:
        pass
    try:
        generated_path.unlink()
    except FileNotFoundError:
        pass

    segment = {
        "start": clip["start"],
        "end": clip["end"],
        "hook": clip["text"][:120].strip(),
        "overall": clip["score"],
        "title": "Preview",
        "hook_enabled": bool(clip.get("hook_enabled")),
        "hook_start": float(clip.get("hook_start") or 0),
        "hook_end": float(clip.get("hook_end") or 0),
        "hook_text": str(clip.get("hook_text") or ""),
        "hook_reason": str(clip.get("hook_reason") or ""),
    }
    segment_file.write_text(
        json.dumps([segment], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        command = [
            sys.executable,
            str(BASE_DIR / "render_clips.py"),
            str(project_dir),
            "--aspect", aspect,
            "--quality", "preview",
            "--segments-file", str(segment_file),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        if result.returncode == 0 and is_valid_mp4(generated_path):
            generated_path.replace(preview_path)
    finally:
        try:
            segment_file.unlink()
        except FileNotFoundError:
            pass
    return result, preview_path


def valid_rendered_mp4s(project_dir):
    """Return only MP4s successfully recorded by the latest render run."""
    output_dir = project_dir / "clips"
    manifest = output_dir / "rendered_clips.json"
    if not manifest.is_file():
        return []
    try:
        names = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    valid = []
    for name in names if isinstance(names, list) else []:
        # Accept filenames only; never allow a manifest entry to escape the project.
        if not isinstance(name, str) or Path(name).name != name:
            continue
        path = output_dir / name
        try:
            if path.suffix.lower() == ".mp4" and path.is_file() and path.stat().st_size > 0:
                valid.append(path)
        except OSError:
            continue
    return valid


def find_source_video(project_dir, transcript_data):
    source = Path(transcript_data.get("video", ""))

    if source.exists():
        return source

    candidates = sorted(project_dir.glob("source.*"))
    return candidates[0] if candidates else None


def render_clip_card(
    clip, index, key_prefix, project_dir, aspect, boundary_context,
    campaign_project=False,
):
    boundary_key = f"boundary_{project_dir.name}_{key_prefix}_{index}"
    if st.session_state.get(boundary_key) not in BOUNDARY_OPTIONS:
        st.session_state[boundary_key] = "Original"
    boundary_mode = st.session_state[boundary_key]
    active_clip = apply_boundary_mode(clip, boundary_mode, boundary_context)
    boundary_variant = active_clip["boundary_variant"]
    hook_available = bool(
        clip.get("automatic_hook")
        and float(clip.get("hook_end") or 0) > float(clip.get("hook_start") or 0)
        and float(active_clip["start"]) <= float(clip.get("hook_start") or 0)
        and float(clip.get("hook_end") or 0) <= float(active_clip["end"])
        and float(clip.get("hook_start") or 0) - float(active_clip["start"]) >= 1.0
    )
    hook_key = f"auto_hook_{project_dir.name}_{key_prefix}_{index}"
    if hook_key not in st.session_state:
        saved_hook = load_hook_settings(project_dir).get(hook_setting_id(clip), {})
        if isinstance(saved_hook, dict) and "hook_enabled" in saved_hook:
            default_hook_enabled = bool(saved_hook["hook_enabled"])
        else:
            default_hook_enabled = bool(campaign_project and hook_available)
        st.session_state[hook_key] = default_hook_enabled
    if not hook_available:
        st.session_state[hook_key] = False
    hook_enabled = bool(st.session_state[hook_key] and hook_available)
    active_clip.update({
        "hook_enabled": hook_enabled,
        "hook_start": float(clip.get("hook_start") or 0),
        "hook_end": float(clip.get("hook_end") or 0),
        "hook_text": str(clip.get("hook_text") or ""),
        "hook_reason": str(clip.get("hook_reason") or ""),
    })
    preview_variant = boundary_variant + ("-auto-hook" if hook_enabled else "-no-hook")
    preview_text = clip["text"]

    if len(preview_text) > 155:
        preview_text = preview_text[:155].rstrip() + "..."

    preview_text = html.escape(preview_text)
    display_title = html.escape(clip.get("title") or f"Clip {index}")
    is_ai_scored = clip.get("scoring_mode") == "ai"
    clip_type = html.escape(str(clip.get("clip_type", "Other")))
    viral_reason = html.escape(str(clip.get("viral_reason", "")))
    rendered_clip, poster = rendered_paths_for_clip(clip, project_dir)
    preview_clip = preview_path_for_clip(
        active_clip, project_dir, aspect, preview_variant
    )
    preview_clicked = False

    with st.container(border=True):
        st.markdown('<span class="clip-row-marker"></span>', unsafe_allow_html=True)
        (
            select_col, thumb_col, content_col, type_col,
            reason_col, score_col, action_col,
        ) = st.columns(
            [0.34, 0.85, 2.8, 0.95, 1.8, 0.68, 1.45],
            vertical_alignment="center"
        )

        with select_col:
            selected = st.checkbox(
                "Select clip",
                key=f"{key_prefix}_{index}",
                label_visibility="collapsed"
            )

        with thumb_col:
            if poster.exists():
                st.image(str(poster), use_container_width=True)
            else:
                st.markdown(
                    '<div class="thumb-placeholder">▶</div>',
                    unsafe_allow_html=True
                )

        with content_col:
            st.markdown(
                f"""
                <div class="clip-title">{display_title}</div>
                <div class="clip-meta">◷ {format_clip_time(active_clip['start'])} → {format_clip_time(active_clip['end'])} &nbsp; {active_clip['duration']:.0f}s</div>
                <p class="clip-preview">“{preview_text}”</p>
                """,
                unsafe_allow_html=True
            )
            hook_widget_enabled = st.checkbox(
                "Auto 2s Hook",
                key=hook_key,
                disabled=not hook_available,
                help=(
                    "Adds the strongest short statement from this clip as a cold open."
                    if hook_available else "No strong standalone hook was found in this clip."
                ),
            )
            if hook_available:
                st.caption(f"Hook: “{clip.get('hook_text', '')}”")
            save_hook_setting(project_dir, clip, hook_widget_enabled and hook_available)

        with type_col:
            if clip.get("dashboard_experimental"):
                st.markdown(
                    '<span class="clip-type-badge">Experimental</span>',
                    unsafe_allow_html=True,
                )
            elif is_ai_scored:
                st.markdown(
                    f'<span class="clip-type-badge">{clip_type}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Local")

        with reason_col:
            if is_ai_scored:
                st.markdown(
                    f'<div class="mini-label">Why this works</div>'
                    f'<p class="clip-why">{viral_reason}</p>',
                    unsafe_allow_html=True,
                )

        with score_col:
            score = int(clip["score"])
            score_class = (
                ""
                if clip.get("dashboard_experimental")
                else "high" if score >= 80 else "medium" if score >= 70 else ""
            )
            score_label = f"{score}/100" if is_ai_scored else str(score)
            st.markdown(
                f'<span class="score-pill {score_class}">{score_label}</span>',
                unsafe_allow_html=True,
            )

        with action_col:
            st.selectbox(
                "Boundary",
                BOUNDARY_OPTIONS,
                key=boundary_key,
            )
            preview_clicked = st.button(
                "▶  Preview",
                key=f"preview_{key_prefix}_{index}",
                use_container_width=True,
            )

            if rendered_clip in valid_rendered_mp4s(project_dir):
                with st.popover("▣  View final", use_container_width=True):
                    st.video(str(rendered_clip))

                with rendered_clip.open("rb") as video_file:
                    st.download_button(
                        "↓  Download",
                        data=video_file,
                        file_name=rendered_clip.name,
                        mime="video/mp4",
                        key=f"clip_download_{key_prefix}_{index}",
                        use_container_width=True
                    )

        if is_ai_scored:
            with st.expander("AI scoring details", expanded=False):
                if clip.get("dashboard_experimental"):
                    st.caption(
                        "Experimental review candidate · "
                        f"{clip.get('decision', 'REJECT')}"
                    )
                st.caption(f"Rationale: {clip.get('viral_reason', '')}")
                st.caption(f"Weakness: {clip.get('ai_weakness', '')}")
                scores = clip.get("ai_scores", {})
                if scores:
                    st.caption(
                        " · ".join([
                            f"Hook {scores.get('hook', 0)}/25",
                            f"Retention {scores.get('retention', 0)}/20",
                            f"Payoff {scores.get('payoff', 0)}/15",
                            "Emotion/Novelty/Conflict "
                            f"{scores.get('emotion_novelty_conflict', 0)}/15",
                            f"Standalone {scores.get('standalone', 0)}/10",
                            f"Shareability {scores.get('shareability', 0)}/10",
                            f"Pacing {scores.get('pacing', 0)}/5",
                        ])
                    )

        if preview_clicked:
            with st.spinner("Rendering preview..."):
                preview_result, preview_clip = render_single_preview(
                    active_clip, project_dir, aspect, preview_variant
                )
            if preview_result.returncode == 0 and is_valid_mp4(preview_clip):
                st.success("Preview ready!")
            else:
                st.error("Preview rendering failed.")
                with st.expander("Advanced details"):
                    st.text(preview_result.stdout)
                    st.text(preview_result.stderr)

        if is_valid_mp4(preview_clip):
            st.caption(
                "Preview · 360×640" if aspect == "9:16" else "Preview · reduced resolution"
            )
            st.video(str(preview_clip))

    return active_clip if selected else None


def show_rendered_clips(project_dir):
    if project_dir is None:
        return

    output_dir = project_dir / "clips"

    if not output_dir.exists():
        return

    mp4_files = sorted(
        valid_rendered_mp4s(project_dir),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not mp4_files:
        return

    st.markdown(
        f'<div class="section-heading">Finished Clips '
        f'<span class="count-pill">{len(mp4_files)}</span></div>',
        unsafe_allow_html=True,
    )
    render_timing = load_project_timings(project_dir).get("final_render")
    if isinstance(render_timing, dict):
        timing_text = (
            f"Render time: {format_runtime(render_timing.get('total_seconds', 0))}"
        )
        average = render_timing.get("average_seconds_per_clip")
        if render_timing.get("selected_clips", 0) > 1 and average is not None:
            timing_text += f" · Average per clip: {format_runtime(average)}"
        st.caption(timing_text)

    cols = st.columns(2, gap="small")

    for index, clip_path in enumerate(mp4_files):
        with cols[index % 2]:
            with st.container(border=True):
                poster_path = clip_path.with_suffix(".png")
                thumb_col, info_col, action_col = st.columns(
                    [0.8, 2.2, 0.9], vertical_alignment="center"
                )
                with thumb_col:
                    if poster_path.is_file():
                        st.image(str(poster_path), use_container_width=True)
                    else:
                        st.markdown(
                            '<div class="thumb-placeholder">▶</div>',
                            unsafe_allow_html=True,
                        )
                with info_col:
                    st.markdown(
                        f'<div class="clip-title">'
                        f'{html.escape(clip_path.stem.replace("-", " ").title())}'
                        f'</div><div class="clip-meta">Final MP4</div>'
                        f'<span class="rendered-pill">Rendered</span>',
                        unsafe_allow_html=True,
                    )
                with action_col:
                    with st.popover("▶", use_container_width=True):
                        st.video(str(clip_path))

                    with open(clip_path, "rb") as video_file:
                        st.download_button(
                            "↓",
                            data=video_file,
                            file_name=clip_path.name,
                            mime="video/mp4",
                            key=f"download_{clip_path.name}",
                            use_container_width=True,
                            help="Download clip",
                        )


def read_json_safely(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def project_created_at(project_dir, metadata=None):
    metadata_created_at = metadata.get("created_at") if isinstance(metadata, dict) else None
    if isinstance(metadata_created_at, str):
        try:
            return datetime.fromisoformat(
                metadata_created_at.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except ValueError:
            pass
    timestamp = project_dir.name.split("_", 1)[0]
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        try:
            return datetime.fromtimestamp(project_dir.stat().st_ctime, timezone.utc)
        except OSError:
            return datetime.fromtimestamp(0, timezone.utc)


def project_summary(project_dir):
    transcript_path = project_dir / "transcript.json"
    transcript = read_json_safely(transcript_path)
    metadata_path = project_dir / "metadata.json"
    metadata = read_json_safely(metadata_path, {})
    metadata_title = metadata.get("title") if isinstance(metadata, dict) else None
    transcript_title = transcript.get("title") if isinstance(transcript, dict) else None
    title = next(
        (
            candidate.strip()
            for candidate in (metadata_title, transcript_title, project_dir.name)
            if isinstance(candidate, str) and candidate.strip()
        ),
        project_dir.name,
    )
    print(f"METADATA PATH: {metadata_path}", flush=True)
    print(f"PROJECT TITLE LOADED: {title}", flush=True)
    source_exists = (project_dir / "source.mp4").is_file()
    transcript_exists = transcript_path.is_file() and isinstance(transcript, dict)
    transcript_segments = transcript.get("segments", []) if transcript_exists else []
    try:
        clip_count = len(build_clip_candidates(transcript_segments))
    except (KeyError, TypeError, ValueError):
        clip_count = 0
    rendered_count = len(valid_rendered_mp4s(project_dir))

    if transcript_exists and clip_count > 0:
        status = "Ready"
    elif transcript_exists:
        status = "Analyzed"
    else:
        status = "Incomplete"

    return {
        "path": project_dir,
        "project_id": project_dir.name,
        "title": title,
        "source_url": metadata.get("source_url") if isinstance(metadata, dict) else None,
        "created_at": project_created_at(project_dir, metadata),
        "source_exists": source_exists,
        "transcript_exists": transcript_exists,
        "clip_count": clip_count,
        "rendered_count": rendered_count,
        "status": status,
    }


def list_project_summaries():
    if not PROJECTS_DIR.is_dir():
        return []
    summaries = [
        project_summary(path)
        for path in PROJECTS_DIR.iterdir()
        if path.is_dir()
    ]
    return sorted(summaries, key=lambda item: item["created_at"], reverse=True)


def history_status(summary):
    if not summary["source_exists"] or not summary["transcript_exists"]:
        return "Incomplete"
    if summary["rendered_count"] > 0:
        return "Ready"
    return "Analyzed"


def compact_source_url(source_url, max_length=58):
    if not isinstance(source_url, str) or not source_url.strip():
        return None
    compact = re.sub(r"^https?://(?:www\.)?", "", source_url.strip(), flags=re.I)
    return compact if len(compact) <= max_length else compact[:max_length - 1] + "…"


def show_projects_view(control_col, workspace_col):
    summaries = list_project_summaries()
    with control_col:
        st.markdown("### Projects")
        st.metric("Total projects", len(summaries))
        st.caption("Newest projects first")

    with workspace_col:
        st.markdown("## Projects")
        st.caption("Open any existing project in the current workspace.")
        if not summaries:
            st.info("No projects found yet.")
            return

        for summary in summaries:
            created_label = summary["created_at"].astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            expander_label = (
                f"{summary['title']}  ·  {summary['status']}  ·  {created_label}"
            )
            with st.expander(expander_label, expanded=False):
                st.caption(f"Project ID: {summary['project_id']}")
                source_col, transcript_col, clips_col, rendered_col, open_col = st.columns(
                    [1, 1, 1, 1, 1.2], vertical_alignment="center"
                )
                source_col.metric("Source", "Yes" if summary["source_exists"] else "No")
                transcript_col.metric(
                    "Transcript", "Yes" if summary["transcript_exists"] else "No"
                )
                clips_col.metric("Candidates", summary["clip_count"])
                rendered_col.metric("Final clips", summary["rendered_count"])
                if open_col.button(
                    "Open project",
                    key=f"open_project_{summary['project_id']}",
                    use_container_width=True,
                ):
                    set_active_project(summary["path"])
                    set_view("home")
                    st.rerun()


def show_history_view(control_col, workspace_col):
    summaries = list_project_summaries()
    with control_col:
        st.markdown("### History")
        st.metric("Projects", len(summaries))
        st.caption("Newest projects first")

    with workspace_col:
        st.markdown("## History")
        st.caption("Chronological project overview · newest first")
        if not summaries:
            st.info("No project history available yet.")
            return
        for summary in summaries:
            status = history_status(summary)
            source_label = compact_source_url(summary.get("source_url"))
            with st.container(border=True):
                info_col, time_col, status_col, clips_col, final_col, open_col = st.columns(
                    [3.2, 1.45, 0.9, 0.75, 0.75, 1.2],
                    vertical_alignment="center",
                )
                with info_col:
                    st.markdown(f"**{summary['title']}**")
                    secondary = f"Project ID: {summary['project_id']}"
                    if source_label:
                        secondary += f" · {source_label}"
                    st.caption(secondary)
                time_col.caption(
                    summary["created_at"].astimezone().strftime("%Y-%m-%d %H:%M:%S")
                )
                status_col.markdown(f"**{status}**")
                clips_col.metric("Candidates", summary["clip_count"])
                final_col.metric("Final", summary["rendered_count"])
                if open_col.button(
                    "Open project",
                    key=f"history_open_{summary['project_id']}",
                    use_container_width=True,
                ):
                    set_active_project(summary["path"])
                    set_view("home")
                    st.rerun()


# --------------------------------------------------
# NEW PROJECT
# --------------------------------------------------

nav_col, control_col, workspace_col = st.columns(
    [1.1, 1.9, 5.2],
    gap="small"
)

with nav_col:
    with st.container(border=True):
        st.markdown(
            '<div class="nav-brand"><span class="product-mark">◆</span> mariundjenson</div>',
            unsafe_allow_html=True,
        )
        for view, label in (
            ("home", "⌂   Home"),
            ("projects", "□   Projects"),
            ("history", "◷   History"),
        ):
            if st.button(
                label,
                key=f"nav_{view}",
                type="primary" if st.session_state.current_view == view else "secondary",
                use_container_width=True,
            ):
                set_view(view)
                st.rerun()
        st.button(
            "⚙   Settings",
            key="nav_settings_placeholder",
            use_container_width=True,
            disabled=True,
            help="Settings are not available yet.",
        )
        st.markdown(
            '<div class="nav-foot">Private local workspace</div>',
            unsafe_allow_html=True,
        )

if st.session_state.current_view == "projects":
    show_projects_view(control_col, workspace_col)
    st.stop()

if st.session_state.current_view == "history":
    show_history_view(control_col, workspace_col)
    st.stop()

with control_col:
    with st.container(border=True):
        st.markdown('<div class="new-project-panel"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-heading"><span class="spark">✦</span> New Project</div>',
            unsafe_allow_html=True,
        )
        source_type = st.selectbox(
            "Source Type",
            ["YouTube URL", "Local File", "Google Drive"],
        )
        url = ""
        drive_url = ""
        uploaded_file = None

        if source_type == "YouTube URL":
            url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
        elif source_type == "Local File":
            uploaded_file = st.file_uploader(
                "Campaign video",
                type=["mp4", "mov", "mkv", "webm"],
            )
        else:
            drive_url = st.text_input(
                "Google Drive link",
                placeholder="Paste a direct, publicly accessible Drive download link",
            )
            if drive_url and not is_direct_google_drive_url(drive_url):
                st.info("Download the campaign file from Google Drive and upload it here.")
                uploaded_file = st.file_uploader(
                    "Upload downloaded campaign file",
                    type=["mp4", "mov", "mkv", "webm"],
                    key="drive_fallback_upload",
                )

        format_choice = st.selectbox(
            "Format",
            ["9:16 Short", "16:9 Original"]
        )
        clip_length = st.selectbox(
            "Clip length",
            ["Auto", "20-40 sec", "40-60 sec"]
        )
        language = st.selectbox(
            "Language",
            ["Auto", "German", "English"]
        )

        ai_available = bool(os.environ.get("OPENAI_API_KEY"))
        ai_status_title = "AI scoring enabled" if ai_available else "Local scoring"
        ai_status_text = (
            "Analyzing for viral potential"
            if ai_available
            else "AI key not configured"
        )
        ai_status_icon = "✓" if ai_available else "○"
        st.markdown(
            f"""
            <div class="ai-status-card">
                <span style="color:#72e6a0;">✦</span>
                <div><strong>{ai_status_title}</strong><small>{ai_status_text}</small></div>
                <span class="ai-check">{ai_status_icon}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Advanced"):
            viral_shorts = st.checkbox(
                "Viral Shorts",
                value=True,
                help="Prioritize publishable short-form virality over general informational value.",
            )
            two_second_hook = st.checkbox(
                "2 second hook at beginning",
                value=True,
                help="Allow a strong line from the same clip to become a short cold open.",
            )
            candidate_debug_frames = st.checkbox(
                "Save candidate debug frames",
                value=True,
                help="Save three small local frames per candidate for analysis debugging.",
            )
            model = st.selectbox(
                "Whisper model",
                ["tiny", "base", "small", "medium", "large-v3"],
                index=2
            )

        with st.expander("Campaign Rules"):
            campaign_name = st.text_input("Campaign Name")
            minimum_views = st.number_input(
                "Minimum Views",
                min_value=0,
                value=0,
                step=1000,
            )
            target_platforms = st.multiselect(
                "Target Platforms",
                ["TikTok", "YouTube Shorts", "Instagram Reels", "Facebook Reels"],
            )
            content_language = st.selectbox(
                "Content Language",
                ["", "English", "German", "Spanish", "French", "Other"],
            )
            required_hashtags = st.text_input(
                "Required Hashtags",
                placeholder="#campaign, #brand",
            )
            required_mentions = st.text_input(
                "Required Mentions",
                placeholder="@brand, @creator",
            )
            campaign_notes = st.text_area("Notes", height=90)

        generate = st.button(
            "✦  Generate Clips",
            type="primary",
            use_container_width=True
        )
        st.markdown(
            '<div class="local-note">🔒 &nbsp; Your video is processed locally</div>',
            unsafe_allow_html=True,
        )

    progress_slot = st.empty()

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

analysis_result = None

with control_col:
    if generate:
        source_missing = (
            (source_type == "YouTube URL" and not url.strip())
            or (source_type == "Local File" and uploaded_file is None)
            or (
                source_type == "Google Drive"
                and not uploaded_file
                and not is_direct_google_drive_url(drive_url)
            )
        )
        if source_missing:
            if source_type == "YouTube URL":
                st.error("Paste a YouTube link first.")
            elif source_type == "Local File":
                st.error("Upload a campaign video first.")
            else:
                st.error("Download the campaign file from Google Drive and upload it here.")

        else:
            generation_started = time.perf_counter()
            project_dir = create_project_dir()
            set_active_project(project_dir)
            save_campaign(project_dir, {
                "campaign_name": campaign_name.strip(),
                "minimum_views": int(minimum_views),
                "target_platforms": list(target_platforms),
                "content_language": content_language,
                "required_hashtags": split_campaign_values(required_hashtags),
                "required_mentions": split_campaign_values(required_mentions),
                "notes": campaign_notes.strip(),
            })
            save_ranking_options(project_dir, {
                "viral_shorts": viral_shorts,
                "two_second_hook": two_second_hook,
                "candidate_debug_frames": candidate_debug_frames,
            })
            st.session_state["_generation_timing"] = {
                "project_id": project_dir.name,
                "started_perf": generation_started,
                "breakdown": {},
            }

            with progress_slot.container(border=True):
                st.markdown("**3. Progress**")
                st.markdown(
                    f"""
                    <div class="workflow-row"><span class="workflow-dot done"></span>{
                        "Downloading video" if source_type in {"YouTube URL", "Google Drive"}
                        else "Importing video"
                    }</div>
                    <div class="workflow-row"><span class="workflow-dot active"></span>Transcribing</div>
                    <div class="workflow-row"><span class="workflow-dot"></span>Analyzing</div>
                    <div class="workflow-row"><span class="workflow-dot"></span>Finding clips</div>
                    <div class="workflow-row"><span class="workflow-dot"></span>Ready</div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(35)

            download_transcription_started = time.perf_counter()
            if source_type == "YouTube URL":
                command = [
                    sys.executable,
                    str(BASE_DIR / "download_and_transcribe.py"),
                    url,
                    "--workdir",
                    str(project_dir),
                    "--model",
                    model,
                    "--device",
                    "cuda",
                    "--compute-type",
                    "float16"
                ]
                try:
                    analysis_result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        cwd=str(BASE_DIR)
                    )
                except OSError as process_error:
                    analysis_result = subprocess.CompletedProcess(
                        command, 1, stdout="", stderr=str(process_error)
                    )
            else:
                command = ["local-import-and-transcribe", source_type]
                try:
                    if source_type == "Google Drive" and uploaded_file is None:
                        source_video = download_direct_drive_video(drive_url, project_dir)
                    else:
                        import_source_type = (
                            "google_drive_upload"
                            if source_type == "Google Drive"
                            else "local_file"
                        )
                        source_video = save_uploaded_video(
                            uploaded_file,
                            project_dir,
                            source_type=import_source_type,
                        )

                    from download_and_transcribe import transcribe

                    transcribe(
                        source_video,
                        project_dir,
                        model,
                        "cuda",
                        "float16",
                    )
                    analysis_result = subprocess.CompletedProcess(
                        command, 0, stdout="Import and transcription complete.", stderr=""
                    )
                except Exception as import_error:
                    analysis_result = subprocess.CompletedProcess(
                        command,
                        1,
                        stdout="",
                        stderr=str(import_error),
                    )
            st.session_state["_generation_timing"]["breakdown"][
                "download_and_transcription_seconds"
            ] = time.perf_counter() - download_transcription_started
            transcription_timing = load_project_timings(project_dir).get("transcription", {})
            if isinstance(transcription_timing, dict):
                measured_transcription = transcription_timing.get("transcription_seconds")
                if isinstance(measured_transcription, (int, float)):
                    st.session_state["_generation_timing"]["breakdown"][
                        "transcription_seconds"
                    ] = float(measured_transcription)

            if analysis_result.returncode == 0:
                multimodal_started = time.perf_counter()
                multimodal_started_at = datetime.now(timezone.utc).isoformat()
                try:
                    from multimodal_analysis import analyze_project

                    multimodal_result = analyze_project(project_dir)
                    if not isinstance(multimodal_result, dict) or not multimodal_result.get("success"):
                        raise RuntimeError("multimodal analysis returned no successful result")
                    analysis_timings = multimodal_result.get("timings")
                    if isinstance(analysis_timings, dict):
                        st.session_state["_generation_timing"]["breakdown"].update({
                            key: float(value)
                            for key, value in analysis_timings.items()
                            if isinstance(value, (int, float))
                        })
                    print("MULTIMODAL ANALYSIS ACTIVE", flush=True)
                except Exception as multimodal_error:
                    try:
                        ensure_failed_analysis_features(
                            project_dir,
                            multimodal_error,
                            multimodal_started_at,
                            time.perf_counter() - multimodal_started,
                        )
                    except OSError as debug_write_error:
                        print(
                            "MULTIMODAL DEBUG ARTIFACT ERROR: "
                            f"{type(debug_write_error).__name__}: {debug_write_error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    print(
                        "MULTIMODAL ANALYSIS FALLBACK: "
                        f"{type(multimodal_error).__name__}: {multimodal_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                finally:
                    st.session_state["_generation_timing"]["breakdown"][
                        "multimodal_analysis_seconds"
                    ] = time.perf_counter() - multimodal_started
                with progress_slot.container(border=True):
                    st.markdown("**3. Progress**")
                    st.markdown(
                        """
                        <div class="workflow-row"><span class="workflow-dot done"></span>Downloading video</div>
                        <div class="workflow-row"><span class="workflow-dot done"></span>Transcribing</div>
                        <div class="workflow-row"><span class="workflow-dot done"></span>Analyzing</div>
                        <div class="workflow-row"><span class="workflow-dot done"></span>Finding clips</div>
                        <div class="workflow-row"><span class="workflow-dot done"></span>Ready</div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(100)
            else:
                complete_generation_timing(project_dir, False)
                st.error("Processing failed.")
                with st.expander("Advanced details"):
                    st.text(analysis_result.stdout)
                    st.text(analysis_result.stderr)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

project_dir = get_project_dir()
transcript_data = load_transcript(project_dir)

with control_col:
    if transcript_data and not generate:
        with progress_slot.container(border=True):
            st.markdown("**3. Progress**")
            st.markdown(
                """
                <div class="workflow-row"><span class="workflow-dot done"></span>Downloading video</div>
                <div class="workflow-row"><span class="workflow-dot done"></span>Transcribing</div>
                <div class="workflow-row"><span class="workflow-dot done"></span>Analyzing</div>
                <div class="workflow-row"><span class="workflow-dot done"></span>Finding clips</div>
                <div class="workflow-row"><span class="workflow-dot done"></span>Ready</div>
                """,
                unsafe_allow_html=True
            )

with workspace_col:
    if not transcript_data:
        st.markdown(
            """
                <div class="empty-workspace">
                <div class="eyebrow">Video repurposing workspace</div>
                <h1>Create clips from your video</h1>
                <p style="color:#a7a7b5;max-width:650px;line-height:1.6;margin:0;">
                    Add a YouTube link, local video, or campaign file to find the strongest moments, review clip suggestions,
                    and export ready-to-share videos from one workspace.
                </p>
                <div class="support-grid">
                    <div class="support-card"><strong>1 · Add video</strong><span>Use YouTube, a local file, or a direct Drive download.</span></div>
                    <div class="support-card"><strong>2 · Review clips</strong><span>Choose from your top suggestions.</span></div>
                    <div class="support-card"><strong>3 · Export</strong><span>Render vertical or original format.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        segments = transcript_data.get("segments", [])
        clips, ai_scoring_active = ranked_clip_candidates(
            project_dir, segments, transcript_data.get("words", [])
        )
        complete_generation_timing(project_dir, True)
        project_name = project_dir.name
        metadata = read_json_safely(project_dir / "metadata.json", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        campaign = read_json_safely(project_dir / "campaign.json", {})
        campaign = campaign if isinstance(campaign, dict) else {}
        campaign_project = any(
            bool(campaign.get(field))
            for field in (
                "campaign_name", "minimum_views", "target_platforms", "content_language",
                "required_hashtags", "required_mentions", "notes",
            )
        )
        current_title = next(
            (
                value.strip()
                for value in (metadata.get("title"), transcript_data.get("title"), project_name)
                if isinstance(value, str) and value.strip()
            ),
            project_name,
        )
        source_url = compact_source_url(metadata.get("source_url"))
        language_label = transcript_data.get("language") or language
        duration_seconds = float(transcript_data.get("duration") or 0)
        if duration_seconds <= 0 and segments:
            duration_seconds = max(float(segment.get("end", 0)) for segment in segments)
        duration_label = (
            f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}"
            if duration_seconds > 0
            else None
        )
        scoring_label = "AI scoring · GPT-5 mini" if ai_scoring_active else "Local scoring"

        st.markdown(
            f"""
            <div class="project-bar">
                <div>
                    <div class="workspace-label">Current Project</div>
                    <h2 class="project-title"><span style="color:#ef4444;">▶</span>&nbsp;
                    {html.escape(current_title)}</h2>
                </div>
                <div style="white-space:nowrap;">
                    <span class="ready-pill">● &nbsp;Ready</span>
                    <span class="ai-pill">✧ &nbsp;{html.escape(scoring_label)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        campaign_summary = []
        if campaign.get("campaign_name"):
            campaign_summary.append(f"Campaign: {campaign['campaign_name']}")
        try:
            campaign_minimum_views = int(campaign.get("minimum_views") or 0)
        except (TypeError, ValueError):
            campaign_minimum_views = 0
        if campaign_minimum_views > 0:
            campaign_summary.append(
                f"Minimum Views: {format_compact_number(campaign_minimum_views)}"
            )
        if campaign.get("content_language"):
            campaign_summary.append(f"Language: {campaign['content_language']}")
        if campaign_summary:
            st.caption(" · ".join(campaign_summary))

        generation_timing = load_project_timings(project_dir).get("generation")
        if isinstance(generation_timing, dict):
            st.caption(
                f"Generation time: {format_runtime(generation_timing.get('total_seconds', 0))}"
            )

        source_video = find_source_video(project_dir, transcript_data)
        st.markdown(
            '<div class="section-heading">Original Video</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            player_col, info_col = st.columns([3.5, 1.9], vertical_alignment="center")
            with player_col:
                if source_video:
                    st.video(str(source_video))
                else:
                    st.info("The original video preview is unavailable.")
            with info_col:
                info_rows = [
                    ("Title", current_title),
                    ("Duration", duration_label),
                    ("Language", str(language_label).title() if language_label else None),
                    ("Source", source_url),
                ]
                info_html = "".join(
                    f'<div class="video-info-row"><span>{html.escape(label)}</span>'
                    f'<span>{html.escape(str(value))}</span></div>'
                    for label, value in info_rows if value
                )
                transcript_excerpt = " ".join(
                    str(segment.get("text", "")).strip() for segment in segments[:4]
                ).strip()
                if len(transcript_excerpt) > 280:
                    transcript_excerpt = transcript_excerpt[:280].rstrip() + "…"
                if transcript_excerpt:
                    info_html += (
                        '<p class="clip-preview" style="margin-top:0.7rem;'
                        '-webkit-line-clamp:4;">'
                        f'{html.escape(transcript_excerpt)}</p>'
                    )
                st.markdown(
                    f'<div class="video-info">{info_html}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("View transcript", expanded=False):
            full_text = ""
            for segment in segments:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                full_text += f"[{start:.1f}s - {end:.1f}s] {text}\n\n"

            st.text_area(
                "Transcript",
                value=full_text,
                height=220,
                label_visibility="collapsed"
            )

        if clips:
            st.markdown(
                '<div class="section-heading">Best Clips <span style="color:#7f8a9a;">✧</span></div>',
                unsafe_allow_html=True,
            )

            selected_clips = []
            visible_clips = clips[:5]
            extra_clips = clips[5:10]

            preview_aspect = "9:16" if format_choice == "9:16 Short" else "16:9"
            boundary_context = build_boundary_context(project_dir, transcript_data)

            for index, clip in enumerate(visible_clips, start=1):
                selected_clip = render_clip_card(
                    clip, index, "main", project_dir, preview_aspect,
                    boundary_context, campaign_project,
                )
                if selected_clip:
                    selected_clips.append(selected_clip)

            if extra_clips:
                with st.expander("Show more clips (6–10)"):
                    for index, clip in enumerate(extra_clips, start=6):
                        selected_clip = render_clip_card(
                            clip, index, "extra", project_dir, preview_aspect,
                            boundary_context, campaign_project,
                        )
                        if selected_clip:
                            selected_clips.append(selected_clip)

            with st.container(border=True):
                action_col1, action_col2 = st.columns([2, 1], vertical_alignment="center")
                with action_col1:
                    st.markdown(f"**{len(selected_clips)} clips selected**")
                    total_duration = sum(float(clip["duration"]) for clip in selected_clips)
                    st.caption(
                        f"Total duration: {int(total_duration // 60)}m "
                        f"{int(total_duration % 60):02d}s"
                    )
                with action_col2:
                    render_button = st.button(
                        "♢  Render selected clips",
                        type="primary",
                        use_container_width=True,
                        disabled=not selected_clips
                    )

            if render_button:
                save_segments(selected_clips, project_dir)
                aspect = "9:16" if format_choice == "9:16 Short" else "16:9"

                render_started = time.perf_counter()
                with st.spinner("Rendering clips..."):
                    render_command = [
                        sys.executable,
                        str(BASE_DIR / "render_clips.py"),
                        str(project_dir),
                        "--aspect",
                        aspect,
                        "--quality",
                        "final"
                    ]
                    try:
                        render_result = subprocess.run(
                            render_command,
                            capture_output=True,
                            text=True,
                            cwd=str(BASE_DIR)
                        )
                    except OSError as process_error:
                        render_result = subprocess.CompletedProcess(
                            render_command, 1, stdout="", stderr=str(process_error)
                        )
                render_seconds = max(0.0, time.perf_counter() - render_started)
                successful_clips = sum(
                    is_valid_mp4(rendered_paths_for_clip(clip, project_dir)[0])
                    for clip in selected_clips
                )
                average_seconds = (
                    render_seconds / successful_clips if successful_clips else None
                )
                render_timing = {
                    "total_seconds": round(render_seconds, 3),
                    "final_render_seconds": round(render_seconds, 3),
                    "success": (
                        render_result.returncode == 0
                        and successful_clips == len(selected_clips)
                    ),
                    "selected_clips": len(selected_clips),
                    "successful_clips": successful_clips,
                    "average_seconds_per_clip": (
                        round(average_seconds, 3) if average_seconds is not None else None
                    ),
                    "completed_at": datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                }
                save_project_timing(project_dir, "final_render", render_timing)

                if render_result.returncode == 0:
                    st.success("Rendering complete!")
                else:
                    st.error("Rendering failed.")
                    with st.expander("Advanced details"):
                        st.text(render_result.stdout)
                        st.text(render_result.stderr)
                if successful_clips == 0:
                    st.caption(f"Render time: {format_runtime(render_seconds)}")
        else:
            st.info("No candidate passed the current quality gate.")

        show_rendered_clips(project_dir)
