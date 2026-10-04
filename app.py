import re
import streamlit as st
import time
from dotenv import load_dotenv
# audio_processor and transcribe are only available in local environments with ffmpeg/torch installed.
# They are imported lazily inside the pipeline fallback so the cloud deploy doesn't crash.
from core.summary import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.segmenter import segment_transcript
from core.followup import generate_study_followup
from core.rag_engine import build_rag_chain, ask_question
from core.infographic import generate_infographic
import streamlit.components.v1 as components
from fpdf import FPDF

load_dotenv()

def create_summary_pdf(title, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    safe_title = title.encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(0, 10, safe_title, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    safe_summary = summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, safe_summary)
    return bytes(pdf.output())

st.set_page_config(
    page_title="VidLearn — AI Video Learning Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg:            #0a0a0f;
    --surface:       #12121a;
    --surface-2:     #1a1a26;
    --surface-3:     #22223a;
    --border:        rgba(139,92,246,0.12);
    --border-2:      rgba(139,92,246,0.22);
    --border-3:      rgba(139,92,246,0.35);
    --text:          #f0f0ff;
    --text-2:        #9898b8;
    --text-3:        #5a5a7a;
    --violet:        #8b5cf6;
    --violet-2:      #a78bfa;
    --violet-3:      #c4b5fd;
    --emerald:       #10b981;
    --emerald-2:     #34d399;
    --violet-bg:     rgba(139,92,246,0.07);
    --emerald-bg:    rgba(16,185,129,0.07);
    --emerald-border:rgba(16,185,129,0.2);
    --r:             14px;
    --r-sm:          10px;
    --r-xs:          7px;
    --shadow:        0 4px 24px rgba(0,0,0,0.5);
    --shadow-glow:   0 0 40px rgba(139,92,246,0.12);
    --t:             all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
.stApp > header { display: none !important; }
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem !important;
    max-width: 1300px !important;
    margin: 0 auto !important;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 15% 10%, rgba(139,92,246,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 50% 50% at 85% 85%, rgba(16,185,129,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
.stElementContainer { margin-bottom: 0 !important; }
[data-testid="stVerticalBlock"] > div { gap: 0.65rem !important; }
[data-testid="stHorizontalBlock"] { gap: 1rem !important; align-items: stretch !important; }
[data-testid="stExpander"] { margin-bottom: 0.5rem !important; }
hr { margin: 1.5rem 0 !important; border: none !important; border-top: 1px solid var(--border) !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.03em !important; color: var(--text) !important; }

/* NAV */
.nav-header { display: flex; align-items: center; justify-content: space-between; padding: 0 0 1.5rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }
.nav-logo { display: flex; align-items: center; gap: 0.75rem; }
.nav-logo-icon { width: 40px; height: 40px; background: linear-gradient(135deg, var(--violet), var(--violet-2)); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 4px 16px rgba(139,92,246,0.35); flex-shrink: 0; }
.nav-logo-text { font-size: 1.1rem; font-weight: 800; letter-spacing: -0.04em; background: linear-gradient(135deg, #fff 30%, var(--violet-3)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-logo-sub { font-size: 0.7rem; color: var(--text-3); font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; display: block; }
.nav-pills { display: flex; gap: 0.35rem; flex-wrap: wrap; }
.nav-pill { padding: 0.3rem 0.8rem; border: 1px solid var(--border); border-radius: 999px; font-size: 0.72rem; font-weight: 500; color: var(--text-2) !important; background: var(--surface); letter-spacing: 0.01em; }
.nav-pill-accent { border-color: var(--border-2); color: var(--violet-3) !important; background: var(--violet-bg); }

/* HERO */
.hero-zone { text-align: center; padding: 2.5rem 1rem 1.5rem; max-width: 720px; margin: 0 auto; }
.hero-eyebrow { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.9rem; border: 1px solid var(--border-2); border-radius: 999px; font-size: 0.72rem; font-weight: 600; color: var(--violet-3); background: var(--violet-bg); margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase; }
.hero-title { font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 900; letter-spacing: -0.05em; line-height: 1.1; margin-bottom: 0.75rem; background: linear-gradient(160deg, #ffffff 0%, #c4b5fd 50%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-sub { font-size: 1rem; color: var(--text-2); line-height: 1.65; margin-bottom: 1.75rem; font-weight: 400; }

/* INPUTS */
.stTextInput > div > div > input, .stSelectbox > div > div, textarea { background: var(--surface-2) !important; border: 1px solid var(--border-2) !important; border-radius: var(--r-xs) !important; color: var(--text) !important; font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important; transition: border-color 0.15s, box-shadow 0.15s; }
.stTextInput > div > div > input:focus, textarea:focus { border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.12) !important; }
label { color: var(--text-2) !important; font-size: 0.78rem !important; font-weight: 500 !important; }

/* BUTTONS */
.stButton > button { background: linear-gradient(135deg, var(--violet), var(--violet-2)) !important; color: #fff !important; border: none !important; border-radius: var(--r-xs) !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.875rem !important; padding: 0.6rem 1.4rem !important; transition: var(--t) !important; box-shadow: 0 2px 12px rgba(139,92,246,0.3) !important; letter-spacing: -0.01em !important; white-space: nowrap !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 24px rgba(139,92,246,0.4) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button[kind="secondary"] { background: var(--surface-2) !important; color: var(--text-2) !important; border: 1px solid var(--border) !important; box-shadow: none !important; }
.stButton > button[kind="secondary"]:hover { background: var(--surface-3) !important; color: var(--text) !important; border-color: var(--border-2) !important; transform: none !important; }
[data-testid="stDownloadButton"] > button { background: var(--surface-2) !important; color: var(--text-2) !important; border: 1px solid var(--border) !important; box-shadow: none !important; font-size: 0.8rem !important; padding: 0.4rem 0.9rem !important; }
[data-testid="stDownloadButton"] > button:hover { background: var(--violet-bg) !important; border-color: var(--border-2) !important; color: var(--violet-3) !important; transform: none !important; }

/* CARDS */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 1.25rem 1.4rem; position: relative; overflow: hidden; transition: var(--t); }
.card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(139,92,246,0.2), transparent); }
.card:hover { border-color: var(--border-2); box-shadow: var(--shadow-glow); }
.card-accent { border-color: var(--border-2); background: linear-gradient(135deg, var(--violet-bg), var(--surface)); }
.card-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-3); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.4rem; }
.card-label-accent { color: var(--violet-2) !important; }
.card-label-emerald { color: var(--emerald-2) !important; }
.card-body { font-size: 0.875rem; line-height: 1.75; color: var(--text-2); white-space: pre-wrap; word-wrap: break-word; max-height: 380px; overflow-y: auto; }
.card-title { font-size: 1.15rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text); line-height: 1.35; }
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.65rem; }
.stat-chip { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.01em; }
.chip-violet { background: var(--violet-bg); border: 1px solid var(--border-2); color: var(--violet-3); }
.chip-emerald { background: var(--emerald-bg); border: 1px solid var(--emerald-border); color: var(--emerald-2); }
.chip-neutral { background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2); }

/* PIPELINE */
.pipeline-wrap { display: flex; gap: 0.4rem; flex-wrap: wrap; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); margin: 0.75rem 0; }
.pipe-item { display: flex; align-items: center; gap: 0.4rem; padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; background: var(--surface-2); border: 1px solid var(--border); color: var(--text-3); transition: var(--t); }
.pipe-item.active { background: var(--violet-bg); border-color: var(--border-2); color: var(--violet-3); }
.pipe-item.done { background: var(--emerald-bg); border-color: var(--emerald-border); color: var(--emerald-2); }
.pipe-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: 0.5; flex-shrink: 0; }
.pipe-item.active .pipe-dot { animation: blink 1.2s ease-in-out infinite; opacity: 1; }
.pipe-item.done .pipe-dot { opacity: 1; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

/* VIDEO */
.video-frame { position: relative; width: 100%; padding-top: 56.25%; border-radius: var(--r); overflow: hidden; border: 1px solid var(--border); background: #000; box-shadow: var(--shadow); }
.video-frame iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: none; }
.video-placeholder { aspect-ratio: 16/9; border-radius: var(--r); border: 1px dashed var(--border-2); display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.5rem; color: var(--text-3); font-size: 0.85rem; background: var(--surface-2); }

/* SECTION HEADER */
.sec-head { display: flex; align-items: center; gap: 0.6rem; margin: 1.5rem 0 0.75rem; }
.sec-head-icon { width: 30px; height: 30px; border-radius: 8px; background: var(--violet-bg); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0; }
.sec-head-text { font-size: 0.95rem; font-weight: 700; letter-spacing: -0.025em; color: var(--text); }
.sec-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* CHAPTERS */
.chapter-list { display: flex; flex-direction: column; gap: 0.5rem; }
.chapter-item { display: flex; align-items: flex-start; gap: 0.85rem; padding: 0.85rem 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-sm); transition: var(--t); }
.chapter-item:hover { border-color: var(--border-2); transform: translateX(4px); box-shadow: 0 2px 12px rgba(139,92,246,0.08); }
.chapter-badge { min-width: 28px; height: 28px; border-radius: 7px; background: var(--violet-bg); border: 1px solid var(--border-2); color: var(--violet-3); font-size: 0.72rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.chapter-title { font-size: 0.88rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; line-height: 1.3; }
.chapter-desc { font-size: 0.78rem; color: var(--text-2); line-height: 1.55; }

/* TRANSCRIPT */
.transcript-wrap { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 1.25rem; font-size: 0.82rem; line-height: 1.8; color: var(--text-2); white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow-y: auto; font-family: 'JetBrains Mono', 'Fira Code', monospace; }

/* CHAT */
.chat-container { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
.chat-messages { padding: 1.25rem; max-height: 420px; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; }
.chat-empty { padding: 2.5rem 1.5rem; text-align: center; }
.chat-empty-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.chat-empty-text { color: var(--text-2); font-size: 0.88rem; margin-top: 0.25rem; }
.msg-row { display: flex; }
.msg-right { justify-content: flex-end; }
.msg-left { justify-content: flex-start; gap: 0.55rem; align-items: flex-start; }
.msg-avatar { width: 26px; height: 26px; border-radius: 7px; background: linear-gradient(135deg, var(--violet), var(--violet-2)); display: flex; align-items: center; justify-content: center; font-size: 0.62rem; color: #fff; flex-shrink: 0; margin-top: 2px; font-weight: 700; }
.bubble { padding: 0.65rem 1rem; font-size: 0.875rem; line-height: 1.65; max-width: 80%; }
.bubble-user { background: var(--violet-bg); border: 1px solid var(--border-2); border-radius: 12px 12px 3px 12px; color: var(--text); }
.bubble-bot { background: var(--surface-2); border: 1px solid var(--border); border-radius: 3px 12px 12px 12px; color: var(--text); }

/* EXPANDER */
[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }
[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 600 !important; font-size: 0.875rem !important; }
[data-testid="stExpander"] summary:hover { color: var(--violet-3) !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--text-2) !important; line-height: 1.7 !important; }
[data-testid="stMarkdownContainer"] strong { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] h2 { font-size: 0.9rem !important; font-weight: 700 !important; color: var(--text) !important; margin: 1rem 0 0.35rem !important; padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); }
[data-testid="stAlert"] { border-radius: var(--r-sm) !important; border: 1px solid var(--border) !important; background: var(--surface-2) !important; font-size: 0.875rem !important; }

/* EMPTY STATE */
.empty-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 50vh; text-align: center; padding: 3rem 2rem; }
.empty-glow { width: 90px; height: 90px; border-radius: 22px; background: linear-gradient(135deg, var(--violet-bg), var(--surface-2)); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center; font-size: 2.5rem; margin-bottom: 1.5rem; box-shadow: 0 0 40px rgba(139,92,246,0.12); }
.empty-h { font-size: 2rem; font-weight: 900; letter-spacing: -0.05em; background: linear-gradient(135deg, #fff 30%, var(--violet-3)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.empty-p { font-size: 0.9rem; color: var(--text-2); max-width: 460px; line-height: 1.65; margin-bottom: 1.75rem; }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.6rem; max-width: 600px; width: 100%; margin-bottom: 2rem; }
.feature-item { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 0.85rem 1rem; text-align: left; transition: var(--t); }
.feature-item:hover { border-color: var(--border-2); transform: translateY(-2px); }
.feature-item-icon { font-size: 1.25rem; margin-bottom: 0.4rem; }
.feature-item-title { font-size: 0.78rem; font-weight: 700; color: var(--text); margin-bottom: 0.15rem; }
.feature-item-desc { font-size: 0.7rem; color: var(--text-3); line-height: 1.4; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* RESPONSIVE */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1.25rem !important; }
    .hero-title { font-size: 1.85rem !important; }
    .nav-pills { display: none; }
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 0.85rem !important; }
    .hero-zone { padding: 1rem 0.25rem; }
    .hero-title { font-size: 1.4rem !important; }
    .card { padding: 0.9rem 1rem; }
    .bubble { max-width: 95%; }
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
    .pipeline-wrap { gap: 0.3rem; }
    .pipe-item { font-size: 0.65rem; padding: 0.22rem 0.5rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────────
for k, v in {"result": None, "chat_history": [], "pipeline_done": False, "pipeline_steps": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ──────────────────────────────────────────────────────────────────────
def get_youtube_id(url: str) -> str:
    for pat in [
        r'(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})',
        r'(?:youtu\.be/)([A-Za-z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})',
    ]:
        m = re.search(pat, url)
        if m: return m.group(1)
    return None

def _step_state(k): return st.session_state.pipeline_steps.get(k, "pending")
def update_step(k, s): st.session_state.pipeline_steps[k] = s

def render_pipeline():
    steps = [("audio","🔊","Audio"),("transcript","📝","Transcript"),("title","🏷️","Title"),
              ("summary","📋","Summary"),("extract","🔍","Extract"),("infographic","🎨","Visuals"),
              ("segment","📚","Chapters"),("followup","📄","Study Plan"),("rag","🧠","AI Chat")]
    html = '<div class="pipeline-wrap">'
    for key, icon, label in steps:
        s = _step_state(key)
        cls = f"pipe-item {s}" if s in ("active","done") else "pipe-item"
        chk = " ✓" if s == "done" else ""
        html += f'<div class="{cls}"><span class="pipe-dot"></span>{icon} {label}{chk}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─── NAV ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-header">
  <div class="nav-logo">
    <div class="nav-logo-icon">⚡</div>
    <div>
      <div class="nav-logo-text">VidLearn</div>
      <span class="nav-logo-sub">AI Video Intelligence</span>
    </div>
  </div>
  <div class="nav-pills">
    <span class="nav-pill">🔊 Transcription</span>
    <span class="nav-pill">📋 Summaries</span>
    <span class="nav-pill">📚 Chapters</span>
    <span class="nav-pill">📄 Study Plans</span>
    <span class="nav-pill nav-pill-accent">⚡ Powered by AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── HERO (only on empty state) ───────────────────────────────────────────────────
if not st.session_state.result:
    st.markdown("""
    <div class="hero-zone">
      <div class="hero-eyebrow">✦ AI-Powered Learning Platform</div>
      <div class="hero-title">Turn Any Video Into<br>a Complete Study Guide</div>
      <div class="hero-sub">Paste a YouTube URL and get instant transcription, AI summary, chapter breakdowns, study plans, and a personal AI tutor — in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

# ─── INPUT BAR ────────────────────────────────────────────────────────────────────
with st.container():
    ic, lc, bc = st.columns([6, 2, 1.5], gap="small")
    with ic:
        source = st.text_input("url", placeholder="🔗  Paste YouTube URL — e.g. youtube.com/watch?v=...", label_visibility="collapsed")
    with lc:
        language = st.selectbox("lang", ["English", "Hinglish"], label_visibility="collapsed")
    with bc:
        run_btn = st.button("⚡ Analyse", use_container_width=True)

# ─── PIPELINE RUNNER ──────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or local file path.")
    else:
        st.session_state.update({"pipeline_done": False, "result": None, "chat_history": [], "pipeline_steps": {}})
        progress_ph = st.empty()
        status_ph = st.empty()
        try:
            with progress_ph.container(): st.info("⚡ Analysing your video — this takes about 20–40 seconds…")
            with status_ph.container(): render_pipeline()

            yt_id = get_youtube_id(source)
            transcript = ""
            chunks = []
            audio_data = {}

            if yt_id:
                try:
                    update_step("audio", "active"); update_step("transcript", "active")
                    with status_ph.container(): render_pipeline()
                    from youtube_transcript_api import YouTubeTranscriptApi
                    api = YouTubeTranscriptApi()
                    t_list = api.list(yt_id)
                    try:
                        t = t_list.find_transcript(['en', 'en-US'])
                    except Exception:
                        try:
                            t = t_list.find_generated_transcript(['en'])
                        except Exception:
                            t = next(iter(t_list))
                    if language.lower() != "english" and t.is_translatable:
                        try: t = t.translate('en')
                        except: pass
                    transcript = " ".join([i.text for i in t.fetch()])
                    update_step("audio", "done"); update_step("transcript", "done")
                    with status_ph.container(): render_pipeline()
                except Exception:
                    pass

            if not transcript:
                # Local fallback (requires ffmpeg + torch + yt-dlp — not available on cloud)
                try:
                    from utils.audio_processor import process_input, cleanup_files
                    from core.transcribe import transcribe_all
                    update_step("audio", "active")
                    with status_ph.container(): render_pipeline()
                    audio_data = process_input(source)
                    chunks = audio_data.get("chunks", [])
                    update_step("audio", "done")
                    update_step("transcript", "active")
                    with status_ph.container(): render_pipeline()
                    transcript = transcribe_all(chunks, translate=(language.lower() != "english"))
                    update_step("transcript", "done")
                except ImportError:
                    raise Exception("This video has no captions available. Local audio transcription is not supported in the cloud version. Please use a YouTube video with captions enabled.")

            def run_step(key, fn):
                update_step(key, "active")
                with status_ph.container(): render_pipeline()
                result = fn()
                update_step(key, "done")
                return result

            title           = run_step("title",      lambda: generate_title(transcript))
            summary         = run_step("summary",     lambda: summarize(transcript))
            update_step("extract", "active")
            with status_ph.container(): render_pipeline()
            study_tasks  = extract_action_items(transcript)
            concepts     = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            update_step("extract", "done")
            infographic_code = run_step("infographic", lambda: generate_infographic(summary, concepts))
            segments         = run_step("segment",     lambda: segment_transcript(transcript))
            study_followup   = run_step("followup",    lambda: generate_study_followup(transcript))
            rag_chain        = run_step("rag",         lambda: build_rag_chain(transcript))

            try:
                cleanup_files(chunks)
                if "original_wav" in audio_data: cleanup_files([audio_data["original_wav"]])
            except (NameError, UnboundLocalError):
                pass  # No local files to clean up on cloud

            with status_ph.container(): render_pipeline()
            st.session_state.result = {
                "title": title, "transcript": transcript, "summary": summary,
                "study_tasks": study_tasks, "core_concepts": concepts,
                "infographic": infographic_code, "open_questions": questions,
                "segments": segments, "study_followup": study_followup,
                "rag_chain": rag_chain, "source": source,
            }
            st.session_state.pipeline_done = True
            progress_ph.success("✅ Analysis complete!")
            time.sleep(0.5); progress_ph.empty(); status_ph.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","infographic","segment","followup","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_ph.error(f"❌ Pipeline failed: {e}")

# ─── RESULTS DASHBOARD ────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    src = r.get("source", "")
    vid_id = get_youtube_id(src) if ("youtube.com" in src or "youtu.be" in src) else None
    word_count = len(r["transcript"].split())

    # Title banner
    st.markdown(f"""
    <div class="card card-accent" style="margin-bottom:1rem;">
      <div class="card-label card-label-accent">⚡ ANALYSIS COMPLETE</div>
      <div class="card-title">{r['title']}</div>
      <div class="stat-row">
        <span class="stat-chip chip-violet">📝 {word_count:,} words</span>
        <span class="stat-chip chip-emerald">✅ Study plan ready</span>
        <span class="stat-chip chip-neutral">🧠 AI tutor active</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Video + Summary
    v_col, s_col = st.columns([5, 4], gap="small")
    with v_col:
        if vid_id:
            st.markdown(f"""
            <div class="video-frame">
              <iframe src="https://www.youtube.com/embed/{vid_id}?rel=0&modestbranding=1"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen></iframe>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="video-placeholder"><span style="font-size:2rem;">📁</span><span>Local file — preview not available</span></div>', unsafe_allow_html=True)
    with s_col:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.5rem;">
          <div class="card-label">📋 LESSON SUMMARY</div>
          <div class="card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)
        
        pdf_bytes = create_summary_pdf(r['title'], r['summary'])
        st.download_button(
            "📥 Download PDF",
            data=pdf_bytes,
            file_name="summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Key Insights
    st.markdown('<div class="sec-head"><div class="sec-head-icon">🔍</div><span class="sec-head-text">Key Insights</span></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(f'<div class="card"><div class="card-label card-label-emerald">✅ STUDY TASKS</div><div class="card-body">{r["study_tasks"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="card-label card-label-accent">💡 CORE CONCEPTS</div><div class="card-body">{r["core_concepts"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="card-label">❓ EXPLORE FURTHER</div><div class="card-body">{r["open_questions"]}</div></div>', unsafe_allow_html=True)

    # Concept Map
    st.markdown('<div class="sec-head"><div class="sec-head-icon">🎨</div><span class="sec-head-text">Visual Concept Map</span></div>', unsafe_allow_html=True)
    if r.get("infographic"):
        mermaid_html = f"""
        <style>body{{margin:0;background:transparent;}} .mc{{background:#12121a;padding:1.25rem;border-radius:10px;border:1px solid rgba(139,92,246,0.15);overflow:auto;display:flex;justify-content:center;}} .mermaid svg{{max-width:100%;height:auto;}}</style>
        <div class="mc"><div class="mermaid">{r["infographic"]}</div></div>
        <script type="module">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';mermaid.initialize({{startOnLoad:true,theme:'dark',securityLevel:'loose'}});</script>
        """
        components.html(mermaid_html, height=420, scrolling=True)
    else:
        st.info("No visual concept map was generated for this content.")

    # Chapter Breakdown
    st.markdown('<div class="sec-head"><div class="sec-head-icon">📚</div><span class="sec-head-text">Chapter Breakdown</span></div>', unsafe_allow_html=True)
    segments = r.get("segments", [])
    if segments:
        html = '<div class="chapter-list">'
        for seg in segments:
            html += f'<div class="chapter-item"><div class="chapter-badge">{seg.get("chapter_number","?")}</div><div><div class="chapter-title">{seg.get("title","Chapter")}</div><div class="chapter-desc">{seg.get("description","")}</div></div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:var(--text-3);font-size:0.85rem;padding:0.5rem 0;">No chapters identified.</div>', unsafe_allow_html=True)

    # Full Transcript
    st.markdown('<div class="sec-head"><div class="sec-head-icon">📝</div><span class="sec-head-text">Full Transcript</span></div>', unsafe_allow_html=True)
    with st.expander("View transcript", expanded=False):
        st.markdown(f'<div class="transcript-wrap">{r["transcript"]}</div>', unsafe_allow_html=True)
        _, dc = st.columns([5, 1])
        with dc:
            st.download_button("📥 Download", data=r["transcript"], file_name="transcript.txt", mime="text/plain", use_container_width=True)

    # Study Plan
    st.markdown('<div class="sec-head"><div class="sec-head-icon">📄</div><span class="sec-head-text">Personalized Study Plan</span></div>', unsafe_allow_html=True)
    followup_text = r.get("study_followup", "")
    with st.expander("View full study plan", expanded=True):
        st.markdown(followup_text)
        _, sc = st.columns([5, 1])
        with sc:
            st.download_button("📥 Download", data=followup_text, file_name="study_plan.md", mime="text/markdown", use_container_width=True)

    # Ask the Video
    st.markdown('<div class="sec-head"><div class="sec-head-icon">💬</div><span class="sec-head-text">Ask the Video</span></div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        msgs_html = '<div class="chat-messages">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                msgs_html += f'<div class="msg-row msg-right"><div class="bubble bubble-user">{msg["content"]}</div></div>'
            else:
                msgs_html += f'<div class="msg-row msg-left"><div class="msg-avatar">AI</div><div class="bubble bubble-bot">{msg["content"]}</div></div>'
        msgs_html += '</div>'
        st.markdown(f'<div class="chat-container">{msgs_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container">
          <div class="chat-empty">
            <div class="chat-empty-icon">🤖</div>
            <div style="font-weight:600;color:var(--text);">Ask me anything about this video</div>
            <div class="chat-empty-text">I've read the entire transcript and I'm ready to help you understand it better.</div>
          </div>
        </div>""", unsafe_allow_html=True)

    qc, bc2 = st.columns([7, 1], gap="small")
    with qc:
        user_input = st.text_input("ask_input", placeholder="Ask a question about the video…", label_visibility="collapsed")
    with bc2:
        ask_btn = st.button("Send →", use_container_width=True)

    if ask_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    render_pipeline()

# ─── EMPTY STATE ─────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="empty-wrap">
      <div class="empty-glow">⚡</div>
      <div class="empty-h">Learn Smarter, Not Harder</div>
      <div class="empty-p">Paste any YouTube lecture, tutorial, or educational video URL above. VidLearn will instantly transcribe, summarize, and create a complete study guide — powered by AI.</div>
      <div class="feature-grid">
        <div class="feature-item"><div class="feature-item-icon">📝</div><div class="feature-item-title">Auto Transcription</div><div class="feature-item-desc">Full text from any YouTube video in seconds</div></div>
        <div class="feature-item"><div class="feature-item-icon">📋</div><div class="feature-item-title">AI Summary</div><div class="feature-item-desc">Concise overview of every key point</div></div>
        <div class="feature-item"><div class="feature-item-icon">📚</div><div class="feature-item-title">Chapter Breakdown</div><div class="feature-item-desc">Topics organized into digestible sections</div></div>
        <div class="feature-item"><div class="feature-item-icon">🎨</div><div class="feature-item-title">Concept Maps</div><div class="feature-item-desc">Visual diagrams of ideas and connections</div></div>
        <div class="feature-item"><div class="feature-item-icon">📄</div><div class="feature-item-title">Study Plans</div><div class="feature-item-desc">Personalized action plan to master the topic</div></div>
        <div class="feature-item"><div class="feature-item-icon">🤖</div><div class="feature-item-title">AI Tutor Chat</div><div class="feature-item-desc">Ask any question about the video content</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
