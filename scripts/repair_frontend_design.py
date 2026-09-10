from pathlib import Path

path = Path('frontend/app.py')
text = path.read_text(encoding='utf-8')
start = text.index('st.markdown(\n    """\n    <style>')
end_marker = '    </style>\n    """,\n    unsafe_allow_html=True,\n)'
end = text.index(end_marker, start) + len(end_marker)

css = '''st.markdown(
    """
    <style>
    :root {
        --cg-ink: #183021;
        --cg-muted: #5d6d63;
        --cg-green: #2f7046;
        --cg-green-dark: #215c37;
        --cg-surface: rgba(255,255,255,.80);
        --cg-border: rgba(255,255,255,.78);
        --cg-shadow: 0 12px 40px rgba(27,63,39,.10);
        --cg-radius: 20px;
    }

    html, body {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif !important;
    }

    body { background: #edf3ee !important; }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 0%, rgba(255,255,255,.95), transparent 34%),
            radial-gradient(circle at 92% 82%, rgba(196,228,202,.42), transparent 32%),
            linear-gradient(145deg, #f5f9f5 0%, #eaf2ec 100%) !important;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    .main .block-container {
        width: min(100%, 1180px);
        max-width: 1180px;
        padding: 2.5rem 2rem 4rem;
    }

    h1, h2, h3 {
        color: var(--cg-ink) !important;
        font-weight: 800 !important;
        letter-spacing: -.035em !important;
    }

    h1 {
        font-size: clamp(2.4rem, 6vw, 4.4rem) !important;
        line-height: 1.02 !important;
        margin: 0 0 .5rem !important;
    }

    h2, h3 { line-height: 1.15 !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--cg-muted); }

    [data-testid="stSidebar"] > div:first-child {
        background: rgba(245,249,245,.94) !important;
        border-right: 1px solid rgba(48,92,61,.10) !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--cg-surface) !important;
        border: 1px solid var(--cg-border) !important;
        border-radius: var(--cg-radius) !important;
        box-shadow: var(--cg-shadow) !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,.74) !important;
        border: 1px solid rgba(255,255,255,.84) !important;
        border-radius: 18px !important;
        padding: 1rem 1.05rem !important;
        box-shadow: 0 8px 28px rgba(27,63,39,.08) !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: var(--cg-ink) !important;
    }

    [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
        background: rgba(255,255,255,.72) !important;
        border: 1px solid rgba(255,255,255,.88) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 28px rgba(27,63,39,.07) !important;
    }

    [data-testid="stFileUploaderDropzone"] { padding: 1rem !important; }

    .stTextInput input, .stSelectbox [data-baseweb="select"] > div {
        min-height: 2.9rem;
        background: rgba(255,255,255,.84) !important;
        border: 1px solid rgba(49,91,61,.13) !important;
        border-radius: 14px !important;
    }

    .stButton > button, .stDownloadButton > button {
        min-height: 2.9rem;
        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,.88) !important;
        padding: .6rem 1.15rem !important;
        font-weight: 700 !important;
        color: #173621 !important;
        background: rgba(255,255,255,.86) !important;
        box-shadow: 0 7px 22px rgba(35,82,49,.10), inset 0 1px 0 rgba(255,255,255,.96) !important;
        transition: box-shadow .2s ease, background .2s ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: rgba(255,255,255,.98) !important;
        box-shadow: 0 10px 28px rgba(35,82,49,.15), inset 0 1px 0 rgba(255,255,255,1) !important;
    }

    .stButton > button:active, .stDownloadButton > button:active { transform: scale(.98); }

    .stButton > button[kind="primary"] {
        color: #fff !important;
        background: linear-gradient(135deg, var(--cg-green), var(--cg-green-dark)) !important;
        border-color: rgba(255,255,255,.28) !important;
        box-shadow: 0 10px 28px rgba(38,105,58,.25) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #347a4b, #28663d) !important;
    }

    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,.78) !important;
        box-shadow: 0 12px 38px rgba(27,63,39,.12) !important;
    }

    [data-testid="stDataFrame"], [data-testid="stAlert"] {
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,.78) !important;
        box-shadow: 0 8px 28px rgba(27,63,39,.08) !important;
    }

    hr {
        border: 0 !important;
        border-top: 1px solid rgba(40,76,50,.12) !important;
        margin: 2rem 0 !important;
    }

    [data-testid="stDialog"] > div {
        width: min(92vw, 520px) !important;
        background: rgba(248,251,248,.97) !important;
        border: 1px solid rgba(255,255,255,.95) !important;
        border-radius: 24px !important;
        box-shadow: 0 24px 70px rgba(24,55,34,.20) !important;
    }

    [data-testid="stDialog"] h2 { font-size: 1.65rem !important; }

    @media (max-width: 900px) {
        .main .block-container { padding: 1.75rem 1.15rem 3rem; }
    }

    @media (max-width: 600px) {
        .main .block-container { padding: 1.25rem .8rem 2.5rem; }
        h1 { font-size: 2.55rem !important; }
        h2 { font-size: 1.55rem !important; }
        h3 { font-size: 1.25rem !important; }
        [data-testid="stMetric"] { padding: .8rem !important; }
        .stButton > button, .stDownloadButton > button {
            min-height: 2.75rem;
            padding: .55rem .9rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)'''

path.write_text(text[:start] + css + text[end:], encoding='utf-8')
print('frontend CSS repaired')
