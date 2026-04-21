"""
Prompt Diff Visualizer — Streamlit UI
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from core.config import get_config
from core.ollama_client import OllamaClient
from core.pipeline import run_diff
from core.schemas import DiffReport, DiffTag, ShiftLevel, AnatomyType
from core.differ import summarise_anatomy_changes

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Diff Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS — dark professional + vibrant
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }

/* ── Header bar ── */
.pdv-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1117 100%);
    border-bottom: 2px solid #6c63ff;
    padding: 1.5rem 2rem 1rem;
    margin: -1rem -1rem 1.5rem;
}
.pdv-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #fff !important;
    margin: 0 0 0.25rem;
    letter-spacing: -0.5px;
}
.pdv-header p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0;
}
.pdv-badge {
    display: inline-block;
    background: #6c63ff22;
    border: 1px solid #6c63ff55;
    color: #a78bfa;
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 20px;
    margin-right: 6px;
    font-weight: 500;
}

/* ── Section headings ── */
.pdv-section {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.02em;
    border-left: 3px solid #6c63ff;
    padding-left: 10px;
    margin: 1.6rem 0 0.8rem;
}

/* ── Cards ── */
.pdv-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.pdv-card-accent {
    background: #1a1f2e;
    border: 1px solid #6c63ff44;
    border-left: 3px solid #6c63ff;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

/* ── Shift badges ── */
.shift-trivial  { background:#0d2818; border-left:4px solid #10b981; color:#6ee7b7; }
.shift-minor    { background:#0c1f38; border-left:4px solid #3b82f6; color:#93c5fd; }
.shift-moderate { background:#2d2208; border-left:4px solid #f59e0b; color:#fcd34d; }
.shift-major    { background:#2d1608; border-left:4px solid #f97316; color:#fdba74; }
.shift-fundamental { background:#2d0a0a; border-left:4px solid #ef4444; color:#fca5a5; }
.shift-badge {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.9em;
}
.shift-badge .shift-title {
    font-size: 1.05em;
    font-weight: 700;
    margin-bottom: 4px;
}
.shift-badge .shift-meta {
    font-size: 0.82em;
    opacity: 0.85;
}

/* ── Diff panels ── */
.diff-panel {
    background: #161b27;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.88em;
    line-height: 2;
    min-height: 80px;
}
.diff-added   { background:#0d2818; border-radius:4px; padding:1px 5px; color:#6ee7b7; }
.diff-removed { background:#2d0a0a; border-radius:4px; padding:1px 5px; color:#fca5a5; text-decoration:line-through; }
.diff-changed { background:#2d2208; border-radius:4px; padding:1px 5px; color:#fcd34d; }
.anat-pill {
    font-size:0.62em; border-radius:4px; padding:1px 5px;
    margin-left:4px; vertical-align:middle; font-weight:600;
    letter-spacing:0.03em;
}

/* ── Model metric cards ── */
.model-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: #6c63ff; }
.model-name { font-weight: 700; font-size: 0.9em; color: #e2e8f0; margin-bottom: 4px; }
.model-shift { font-size: 0.8em; margin-top: 3px; color: #94a3b8; }

/* ── Output boxes ── */
.output-box {
    background: #161b27;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.85em;
    white-space: pre-wrap;
    max-height: 280px;
    overflow-y: auto;
    color: #cbd5e0;
    line-height: 1.6;
}
.output-error {
    background: #2d0a0a;
    border: 1px solid #ef444466;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.85em;
    color: #fca5a5;
    font-family: monospace;
}

/* ── Judge verdict ── */
.verdict-card {
    background: #0c1f38;
    border: 1px solid #3b82f655;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 8px;
}
.verdict-italic {
    font-style: italic;
    color: #93c5fd;
    font-size: 0.92em;
    line-height: 1.6;
}

/* ── Anatomy pills ── */
.anat-role        { background:#4c1d95; color:#ddd6fe; }
.anat-instruction { background:#1e3a5f; color:#bfdbfe; }
.anat-constraint  { background:#7f1d1d; color:#fecaca; }
.anat-example     { background:#14532d; color:#bbf7d0; }
.anat-format      { background:#713f12; color:#fde68a; }
.anat-context     { background:#134e4a; color:#99f6e4; }
.anat-unknown     { background:#1f2937; color:#9ca3af; }

/* ── Streamlit overrides ── */
.stButton>button {
    background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity: 0.88 !important; }
.stButton>button:disabled { background: #2d3748 !important; color: #64748b !important; }

.stTextArea textarea {
    background: #161b27 !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.88em !important;
}
.stTextArea textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px #6c63ff33 !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #161b27 !important;
    border-color: #2d3748 !important;
    color: #e2e8f0 !important;
}

div[data-testid="stExpander"] {
    background: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"]:hover { border-color: #6c63ff55 !important; }

.stProgress > div > div { background: #6c63ff !important; }

/* tab-like section labels */
.stMarkdown h3 { color: #e2e8f0 !important; }
hr { border-color: #2d3748 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Colour / style maps
# ─────────────────────────────────────────────
SHIFT_CSS = {
    ShiftLevel.TRIVIAL:     ("shift-trivial",     "✅", "#10b981"),
    ShiftLevel.MINOR:       ("shift-minor",       "🔵", "#3b82f6"),
    ShiftLevel.MODERATE:    ("shift-moderate",    "🟡", "#f59e0b"),
    ShiftLevel.MAJOR:       ("shift-major",       "🟠", "#f97316"),
    ShiftLevel.FUNDAMENTAL: ("shift-fundamental", "🔴", "#ef4444"),
}

ANAT_CSS = {
    AnatomyType.ROLE:        "anat-role",
    AnatomyType.INSTRUCTION: "anat-instruction",
    AnatomyType.CONSTRAINT:  "anat-constraint",
    AnatomyType.EXAMPLE:     "anat-example",
    AnatomyType.FORMAT:      "anat-format",
    AnatomyType.CONTEXT:     "anat-context",
    AnatomyType.UNKNOWN:     "anat-unknown",
}

ANAT_BG = {
    AnatomyType.ROLE:        ("#4c1d95", "#ddd6fe"),
    AnatomyType.INSTRUCTION: ("#1e3a5f", "#bfdbfe"),
    AnatomyType.CONSTRAINT:  ("#7f1d1d", "#fecaca"),
    AnatomyType.EXAMPLE:     ("#14532d", "#bbf7d0"),
    AnatomyType.FORMAT:      ("#713f12", "#fde68a"),
    AnatomyType.CONTEXT:     ("#134e4a", "#99f6e4"),
    AnatomyType.UNKNOWN:     ("#1f2937", "#9ca3af"),
}

# ─────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────
@st.cache_resource
def get_ollama_client() -> OllamaClient:
    return OllamaClient(get_config())

@st.cache_data(ttl=30)
def fetch_local_models() -> tuple[list[str], dict]:
    health = get_ollama_client().health_check()
    return health.get("models", []), health

# ─────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────
def render_shift_badge(level: ShiftLevel, similarity: float, shift_score: float, label: str):
    cls, icon, _ = SHIFT_CSS.get(level, ("shift-trivial", "✅", "#10b981"))
    st.markdown(
        f'<div class="shift-badge {cls}">'
        f'<div class="shift-title">{icon} {level.value.upper()} SHIFT</div>'
        f'<div>{label}</div>'
        f'<div class="shift-meta">Similarity: <b>{similarity:.3f}</b> &nbsp;|&nbsp; Shift score: <b>{shift_score:.3f}</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_diff_html(segments) -> str:
    parts = []
    for seg in segments:
        anat_cls = ANAT_CSS.get(seg.anatomy, "anat-unknown")
        anat_pill = (
            f'<span class="anat-pill {anat_cls}">{seg.anatomy}</span>'
            if seg.tag != DiffTag.UNCHANGED else ""
        )
        if seg.tag == DiffTag.ADDED:
            parts.append(f'<span class="diff-added">{seg.text}{anat_pill}</span> ')
        elif seg.tag == DiffTag.REMOVED:
            parts.append(f'<span class="diff-removed">{seg.text}{anat_pill}</span> ')
        elif seg.tag == DiffTag.CHANGED:
            parts.append(f'<span class="diff-changed">{seg.text}{anat_pill}</span> ')
        else:
            parts.append(f'<span style="color:#94a3b8">{seg.text}</span> ')
    return "".join(parts)


def render_output_comparison(result, expanded: bool = False):
    cls, icon, accent = SHIFT_CSS.get(result.score.level, ("shift-trivial", "✅", "#10b981"))
    header = (
        f"{icon} **{result.model_name}** — "
        f"Output shift: `{result.score.shift_score:.3f}` ({result.score.level.value})"
    )
    with st.expander(header, expanded=expanded):
        is_error_v1 = result.output_v1.startswith("[ERROR")
        is_error_v2 = result.output_v2.startswith("[ERROR")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div style="font-size:0.8em;color:#6c63ff;font-weight:600;margin-bottom:6px">▌ OUTPUT v1</div>', unsafe_allow_html=True)
            box_cls = "output-error" if is_error_v1 else "output-box"
            st.markdown(f'<div class="{box_cls}">{result.output_v1}</div>', unsafe_allow_html=True)
            st.caption(f"⏱ {result.latency_v1_ms:.0f} ms" + (f" · {result.tokens_v1} tok" if result.tokens_v1 else ""))
        with col2:
            st.markdown('<div style="font-size:0.8em;color:#a78bfa;font-weight:600;margin-bottom:6px">▌ OUTPUT v2</div>', unsafe_allow_html=True)
            box_cls = "output-error" if is_error_v2 else "output-box"
            st.markdown(f'<div class="{box_cls}">{result.output_v2}</div>', unsafe_allow_html=True)
            st.caption(f"⏱ {result.latency_v2_ms:.0f} ms" + (f" · {result.tokens_v2} tok" if result.tokens_v2 else ""))

        st.markdown(
            f'<div class="shift-badge {cls}" style="margin-top:10px;padding:8px 12px;font-size:0.82em;">'
            f'Semantic similarity: <b>{result.score.similarity:.3f}</b> — {result.score.label}</div>',
            unsafe_allow_html=True,
        )

        if is_error_v1 or is_error_v2:
            st.markdown(
                '<div style="background:#2d0a0a;border:1px solid #ef444444;border-radius:6px;'
                'padding:8px 12px;font-size:0.8em;color:#fca5a5;margin-top:6px;">'
                '⚠️ One or both outputs errored. Check that the model name matches exactly '
                'what Ollama reports (e.g. <code>llama3.1:8b</code> not <code>llama3</code>). '
                'Run <code>ollama list</code> to verify.</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar() -> tuple[list[str], bool, bool]:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;'
            'padding:0.5rem 0 1rem;border-bottom:1px solid #2d3748;margin-bottom:1rem;">'
            '⚙️ Settings</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Checking Ollama..."):
            models, health = fetch_local_models()

        if health["status"] == "online":
            st.markdown(
                f'<div style="background:#0d2818;border:1px solid #10b98155;border-radius:8px;'
                f'padding:8px 12px;font-size:0.82em;color:#6ee7b7;margin-bottom:12px;">'
                f'✅ Ollama online — <b>{health["model_count"]}</b> model(s) found</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error(f"❌ Ollama offline\n\n`{health.get('error','')}`")
            st.info("Start Ollama with: `ollama serve`")
            st.stop()

        if not models:
            st.warning("No models found.\n\nRun: `ollama pull mistral`")
            st.stop()

        st.markdown('<div style="font-size:0.75em;color:#6c63ff;font-weight:600;margin:0.8rem 0 0.3rem;letter-spacing:0.06em">MODELS TO TEST</div>', unsafe_allow_html=True)
        selected = st.multiselect(
            "Models",
            options=models,
            default=models[:min(3, len(models))],
            help="Only locally available models shown. Names are exactly as reported by Ollama.",
            label_visibility="collapsed",
        )

        st.markdown('<div style="font-size:0.75em;color:#6c63ff;font-weight:600;margin:1rem 0 0.3rem;letter-spacing:0.06em">JUDGE</div>', unsafe_allow_html=True)
        run_judge = st.toggle("Run LLM judge", value=True,
                              help="Uses your first available model to explain what semantically changed")

        st.markdown('<div style="font-size:0.75em;color:#6c63ff;font-weight:600;margin:1rem 0 0.3rem;letter-spacing:0.06em">DISPLAY</div>', unsafe_allow_html=True)
        show_raw = st.toggle("Show raw JSON", value=False)

        st.markdown(
            '<div style="font-size:0.72em;color:#475569;border-top:1px solid #2d3748;'
            'padding-top:1rem;margin-top:1.5rem;line-height:1.6;">'
            '🔒 100% local via Ollama.<br>No data leaves your machine.</div>',
            unsafe_allow_html=True,
        )

    return selected, run_judge, show_raw


# ─────────────────────────────────────────────
# Example pairs
# ─────────────────────────────────────────────
EXAMPLES = {
    "Summarization (add role + format)": (
        "Summarize the following article in simple language.",
        "You are a senior technical writer. Summarize the following article for a developer audience. Respond in bullet points. Do not exceed 5 bullets. Avoid marketing language.",
    ),
    "Classification (add constraint)": (
        "Classify the sentiment of this text as positive, negative, or neutral.",
        "Classify the sentiment of this text as positive, negative, or neutral. Do not explain your reasoning. Respond with only one word.",
    ),
    "Roleplay (fundamental rewrite)": (
        "Tell me about climate change.",
        "You are a climate scientist presenting to policymakers. Explain the three most critical near-term risks of climate change with supporting data. Use formal language. Avoid jargon.",
    ),
}

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        '<div class="pdv-header">'
        '<h1>🔍 Prompt Diff Visualizer</h1>'
        '<p>'
        '<span class="pdv-badge">Semantic scoring</span>'
        '<span class="pdv-badge">Anatomy tagging</span>'
        '<span class="pdv-badge">Multi-model</span>'
        '<span class="pdv-badge">LLM judge</span>'
        '<span class="pdv-badge">100% local</span>'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    selected_models, run_judge, show_raw = render_sidebar()

    # ── Input section ──
    st.markdown('<div class="pdv-section">📝 Prompt inputs</div>', unsafe_allow_html=True)

    example_choice = st.selectbox(
        "Load an example pair",
        options=["— custom input —"] + list(EXAMPLES.keys()),
        index=0,
    )
    default_v1, default_v2 = EXAMPLES.get(example_choice, ("", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="font-size:0.8em;color:#6c63ff;font-weight:600;margin-bottom:4px">▌ PROMPT v1 — original</div>', unsafe_allow_html=True)
        prompt_v1 = st.text_area("pv1", value=default_v1, height=180,
                                  label_visibility="collapsed",
                                  placeholder="Paste your original prompt here...")
    with col2:
        st.markdown('<div style="font-size:0.8em;color:#a78bfa;font-weight:600;margin-bottom:4px">▌ PROMPT v2 — revised</div>', unsafe_allow_html=True)
        prompt_v2 = st.text_area("pv2", value=default_v2, height=180,
                                  label_visibility="collapsed",
                                  placeholder="Paste your revised prompt here...")

    with st.expander("📎 Or upload .txt files"):
        uc1, uc2 = st.columns(2)
        with uc1:
            f1 = st.file_uploader("v1 file", type=["txt"], key="f1")
            if f1:
                prompt_v1 = f1.read().decode("utf-8")
                st.caption(f"Loaded: {f1.name}")
        with uc2:
            f2 = st.file_uploader("v2 file", type=["txt"], key="f2")
            if f2:
                prompt_v2 = f2.read().decode("utf-8")
                st.caption(f"Loaded: {f2.name}")

    if not selected_models:
        st.warning("⬅ Select at least one model from the sidebar.")
        return

    run_btn = st.button(
        "▶ Run Diff",
        type="primary",
        disabled=not (prompt_v1.strip() and prompt_v2.strip()),
    )
    if not run_btn:
        return

    # ── Pipeline ──
    with st.spinner("Running pipeline..."):
        prog = st.progress(0, text="Structural diff + embeddings...")
        report: DiffReport = run_diff(
            prompt_v1=prompt_v1.strip(),
            prompt_v2=prompt_v2.strip(),
            models=selected_models,
            run_judge=run_judge,
        )
        prog.progress(100, text="Done ✅")

    st.markdown("---")

    # ── 1. Prompt semantic score ──
    st.markdown('<div class="pdv-section">🧠 Prompt semantic shift</div>', unsafe_allow_html=True)
    ps = report.prompt_semantic.prompt_score
    render_shift_badge(ps.level, ps.similarity, ps.shift_score, ps.label)
    st.caption(f"Embedding model: `{report.embedding_model}`")

    # ── 2. Structural diff ──
    st.markdown('<div class="pdv-section">🔬 Structural diff + anatomy tags</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown('<div style="font-size:0.8em;color:#6c63ff;font-weight:600;margin-bottom:6px">▌ PROMPT v1</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="diff-panel">{render_diff_html(report.structural_diff.segments_v1)}</div>', unsafe_allow_html=True)
    with dc2:
        st.markdown('<div style="font-size:0.8em;color:#a78bfa;font-weight:600;margin-bottom:6px">▌ PROMPT v2</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="diff-panel">{render_diff_html(report.structural_diff.segments_v2)}</div>', unsafe_allow_html=True)

    # Legend
    st.markdown(
        '<div style="font-size:0.78em;margin:6px 0 12px;display:flex;gap:12px;flex-wrap:wrap;">'
        '<span class="diff-added" style="border-radius:4px;padding:1px 8px;">➕ Added</span>'
        '<span class="diff-removed" style="border-radius:4px;padding:1px 8px;">➖ Removed</span>'
        '<span class="diff-changed" style="border-radius:4px;padding:1px 8px;">✏️ Changed</span>'
        '<span style="color:#94a3b8">· pills = anatomy type</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Anatomy summary
    anat_summary = summarise_anatomy_changes(report.structural_diff)
    if anat_summary:
        acols = st.columns(min(len(anat_summary), 4))
        for i, (atype, changes) in enumerate(anat_summary.items()):
            bg, fg = ANAT_BG.get(atype, ("#1f2937", "#9ca3af"))
            with acols[i % len(acols)]:
                items = "".join(
                    f'<div style="font-size:0.78em;margin-top:4px;opacity:0.9">{c}</div>'
                    for c in changes[:3]
                )
                st.markdown(
                    f'<div style="background:{bg};border-radius:8px;padding:10px 12px;'
                    f'margin-bottom:8px;border:1px solid {fg}33;">'
                    f'<div style="color:{fg};font-size:0.75em;font-weight:700;'
                    f'letter-spacing:0.06em">{atype.upper()}</div>{items}</div>',
                    unsafe_allow_html=True,
                )

    # ── 3. Per-model output comparison ──
    st.markdown('<div class="pdv-section">🤖 Per-model output comparison</div>', unsafe_allow_html=True)

    mcols = st.columns(len(report.output_results))
    for i, res in enumerate(report.output_results):
        cls, icon, accent = SHIFT_CSS.get(res.score.level, ("shift-trivial", "✅", "#10b981"))
        with mcols[i]:
            st.markdown(
                f'<div class="model-card">'
                f'<div class="model-name">{res.model_name}</div>'
                f'<div style="color:{accent};font-size:0.82em;font-weight:600">'
                f'{icon} {res.score.level.value.title()}</div>'
                f'<div class="model-shift">shift: <b>{res.score.shift_score:.3f}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    for i, res in enumerate(report.output_results):
        render_output_comparison(res, expanded=(i == 0))

    # ── 4. Judge verdict ──
    if report.verdict:
        st.markdown('<div class="pdv-section">⚖️ LLM judge verdict</div>', unsafe_allow_html=True)
        v = report.verdict

        if v.judge_model == "none" or "unavailable" in v.intent_change.lower():
            st.markdown(
                '<div class="pdv-card" style="border-color:#f59e0b44;">'
                '⚠️ Judge unavailable — no models responded. Check Ollama is running and at least one model is pulled.</div>',
                unsafe_allow_html=True,
            )
        else:
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.markdown(
                    '<div class="pdv-card-accent">'
                    '<div style="font-size:0.72em;color:#a78bfa;font-weight:600;letter-spacing:0.06em;margin-bottom:6px">INTENT CHANGE</div>'
                    f'<div class="verdict-italic">{v.intent_change}</div></div>',
                    unsafe_allow_html=True,
                )
            with vc2:
                st.markdown(
                    '<div style="background:#0d2818;border:1px solid #10b98133;border-left:3px solid #10b981;border-radius:10px;padding:14px 16px;">'
                    '<div style="font-size:0.72em;color:#6ee7b7;font-weight:600;letter-spacing:0.06em;margin-bottom:6px">v2 GAINED</div>'
                    f'<div style="color:#d1fae5;font-size:0.9em;line-height:1.6">{v.gained}</div></div>',
                    unsafe_allow_html=True,
                )
            with vc3:
                st.markdown(
                    '<div style="background:#2d2208;border:1px solid #f59e0b33;border-left:3px solid #f59e0b;border-radius:10px;padding:14px 16px;">'
                    '<div style="font-size:0.72em;color:#fcd34d;font-weight:600;letter-spacing:0.06em;margin-bottom:6px">v2 LOST</div>'
                    f'<div style="color:#fef3c7;font-size:0.9em;line-height:1.6">{v.lost}</div></div>',
                    unsafe_allow_html=True,
                )

            rec_styles = {
                "use_v1":            ("🔙 Use v1", "#2d0a0a", "#ef4444", "#fca5a5"),
                "use_v2":            ("✅ Use v2", "#0d2818", "#10b981", "#6ee7b7"),
                "context_dependent": ("🤔 Context dependent", "#1a1f2e", "#6c63ff", "#a78bfa"),
            }
            rl, rbg, rb, rt = rec_styles.get(v.recommendation,
                ("❓ " + v.recommendation, "#1a1f2e", "#6c63ff", "#a78bfa"))
            st.markdown(
                f'<div style="background:{rbg};border:1px solid {rb}44;border-left:4px solid {rb};'
                f'border-radius:10px;padding:12px 16px;margin-top:10px;">'
                f'<span style="color:{rt};font-weight:700;font-size:0.95em">{rl}</span>'
                f'<span style="color:#64748b;font-size:0.78em;margin-left:12px">Judge: {v.judge_model}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Raw JSON ──
    if show_raw:
        st.markdown('<div class="pdv-section">📄 Raw JSON report</div>', unsafe_allow_html=True)
        st.json(report.model_dump())


if __name__ == "__main__":
    main()
