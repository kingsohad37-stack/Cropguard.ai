"""Streamlit client for genuine CropGuard inference only."""
import os
import requests
import streamlit.components.v1 as components
import streamlit as st
from PIL import Image, UnidentifiedImageError


class _CameraCapture:
    """Small file-like wrapper for a captured camera image."""
    def __init__(self, data, name="camera.jpg", content_type="image/jpeg"):
        self._data = data
        self.name = name
        self.type = content_type

    def getvalue(self):
        return self._data


st.set_page_config(page_title="CropGuard AI", page_icon="🌿", layout="wide")

# Premium visual layer only: no inference, routes, data, or business logic changed.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --cg-ink: #183021;
        --cg-muted: #5e7065;
        --cg-green: #2f6f44;
        --cg-glass: rgba(255,255,255,.58);
        --cg-glass-strong: rgba(255,255,255,.72);
        --cg-border: rgba(255,255,255,.58);
        --cg-shadow: 0 18px 60px rgba(37, 67, 47, .13);
        --cg-shadow-hover: 0 24px 70px rgba(37, 67, 47, .19);
        --cg-radius: 22px;
        --cg-ease: cubic-bezier(.4,0,.2,1);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif;
    }

    body {
        background: #eef3ee !important;
    }

    body::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -2;
        background-image:
            linear-gradient(135deg, rgba(242,248,239,.74), rgba(238,246,243,.54)),
            url('/app/static/cropguard-bg.jpg');
        background-size: cover;
        background-position: center;
        filter: saturate(.92);
        transform: scale(1.035);
    }

    body::after {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        background:
            radial-gradient(circle at 15% 10%, rgba(255,255,255,.78), transparent 30%),
            radial-gradient(circle at 90% 75%, rgba(180,225,187,.28), transparent 34%),
            rgba(247,250,247,.22);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }

    .block-container {
        max-width: 1180px;
        padding: 3.5rem 2rem 4rem;
    }

    h1, h2, h3 {
        color: var(--cg-ink) !important;
        letter-spacing: -.045em !important;
        font-weight: 800 !important;
    }

    h1 {
        font-size: clamp(2.5rem, 5vw, 4.7rem) !important;
        line-height: .98 !important;
        margin-bottom: .45rem !important;
    }

    h2, h3 {
        line-height: 1.08 !important;
    }

    p, label, .stCaption, [data-testid="stMarkdownContainer"] {
        color: var(--cg-muted);
    }

    /* Glass navigation/sidebar */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(236,245,237,.54) !important;
        backdrop-filter: blur(24px) saturate(140%);
        -webkit-backdrop-filter: blur(24px) saturate(140%);
        border-right: 1px solid rgba(255,255,255,.62);
        box-shadow: 10px 0 40px rgba(38,67,48,.08);
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: .65rem;
    }

    /* Soft glass cards for the app's existing Streamlit sections */
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stMetric"],
    [data-testid="stAlert"],
    [data-testid="stFileUploader"],
    [data-testid="stCameraInput"],
    [data-testid="stDataFrame"] {
        border-radius: var(--cg-radius) !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--cg-glass) !important;
        border: 1px solid var(--cg-border) !important;
        box-shadow: var(--cg-shadow) !important;
        backdrop-filter: blur(20px) saturate(135%);
        -webkit-backdrop-filter: blur(20px) saturate(135%);
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,.47) !important;
        border: 1px solid rgba(255,255,255,.58) !important;
        padding: 1.1rem 1.15rem !important;
        box-shadow: 0 12px 38px rgba(38,67,48,.09) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        transition: transform .22s var(--cg-ease), box-shadow .22s var(--cg-ease);
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: var(--cg-shadow-hover) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--cg-ink) !important;
        font-weight: 800 !important;
        letter-spacing: -.035em;
    }

    /* Inputs / uploader */
    .stTextInput input,
    .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stCameraInput"] > div {
        background: rgba(255,255,255,.46) !important;
        border: 1px solid rgba(255,255,255,.68) !important;
        border-radius: 18px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.7), 0 8px 30px rgba(38,67,48,.07) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 1.5rem !important;
        transition: transform .22s var(--cg-ease), box-shadow .22s var(--cg-ease), background .22s var(--cg-ease);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,.62) !important;
        box-shadow: var(--cg-shadow-hover) !important;
    }

    /* Glossy pill buttons */
    .stButton > button,
    .stDownloadButton > button {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.65) !important;
        border-radius: 999px !important;
        min-height: 3rem;
        padding: .65rem 1.25rem !important;
        font-weight: 700 !important;
        letter-spacing: -.01em;
        color: #173621 !important;
        background: linear-gradient(135deg, rgba(255,255,255,.78), rgba(224,242,228,.48)) !important;
        box-shadow: 0 10px 28px rgba(43,91,55,.14), inset 0 1px 0 rgba(255,255,255,.9) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: all .2s ease !important;
    }

    .stButton > button::before,
    .stDownloadButton > button::before {
        content: "";
        position: absolute;
        top: -80%;
        left: -25%;
        width: 42%;
        height: 230%;
        transform: rotate(25deg);
        background: linear-gradient(90deg, transparent, rgba(255,255,255,.52), transparent);
        opacity: .65;
        pointer-events: none;
        transition: left .45s var(--cg-ease);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: scale(1.035) translateY(-1px);
        box-shadow: 0 16px 38px rgba(43,91,55,.2), inset 0 1px 0 rgba(255,255,255,.95) !important;
    }

    .stButton > button:hover::before,
    .stDownloadButton > button:hover::before {
        left: 110%;
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: scale(.97);
    }

    /* Primary action gets a richer botanical glass treatment */
    .stButton > button[kind="primary"] {
        color: white !important;
        background: linear-gradient(135deg, rgba(53,121,72,.95), rgba(31,92,51,.84)) !important;
        box-shadow: 0 14px 34px rgba(39,104,57,.28), inset 0 1px 0 rgba(255,255,255,.28) !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 20px 44px rgba(39,104,57,.34), inset 0 1px 0 rgba(255,255,255,.34) !important;
    }

    /* Uploaded image / CAM result */
    [data-testid="stImage"] img {
        border-radius: 22px !important;
        border: 1px solid rgba(255,255,255,.62);
        box-shadow: 0 18px 55px rgba(38,67,48,.14);
        transition: transform .3s var(--cg-ease), box-shadow .3s var(--cg-ease);
    }

    [data-testid="stImage"] img:hover {
        transform: translateY(-3px) scale(1.008);
        box-shadow: 0 24px 70px rgba(38,67,48,.2);
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.62) !important;
        box-shadow: var(--cg-shadow) !important;
        background: rgba(255,255,255,.45) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border: 1px solid rgba(255,255,255,.62) !important;
        box-shadow: 0 10px 32px rgba(38,67,48,.08) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    /* Dividers become softer */
    hr {
        border: 0 !important;
        border-top: 1px solid rgba(54,89,64,.12) !important;
        margin: 2.5rem 0 !important;
    }

    /* Startup language modal */
    [data-testid="stDialog"] > div {
        background: rgba(245,250,246,.76) !important;
        border: 1px solid rgba(255,255,255,.72) !important;
        border-radius: 26px !important;
        box-shadow: 0 28px 90px rgba(28,63,40,.22) !important;
        backdrop-filter: blur(26px) saturate(140%);
        -webkit-backdrop-filter: blur(26px) saturate(140%);
    }

    [data-testid="stDialog"] h2 {
        font-size: 1.7rem !important;
    }

    /* Gentle load-in motion */
    .main .block-container > div {
        animation: cgFadeUp .55s var(--cg-ease) both;
    }

    @keyframes cgFadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
    }

    @media (max-width: 700px) {
        .block-container { padding: 2rem 1rem 3rem; }
        h1 { font-size: 2.7rem !important; }
        [data-testid="stMetric"] { padding: .9rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Explicitly stop browser camera tracks when the page/app is backgrounded.
components.html(
    """<script>
    (() => {
      const stopVideos = (root) => {
        try {
          root.querySelectorAll('video').forEach((video) => {
            const stream = video.srcObject;
            if (stream && typeof stream.getTracks === 'function') {
              stream.getTracks().forEach((track) => track.stop());
              video.srcObject = null;
            }
          });
          root.querySelectorAll('iframe').forEach((frame) => {
            try { stopVideos(frame.contentWindow.document); } catch (_) {}
          });
        } catch (_) {}
      };
      const stopCamera = () => {
        try { stopVideos(window.parent.document); } catch (_) {}
        stopVideos(document);
      };
      try { window.parent.document.addEventListener('visibilitychange', () => {
        if (window.parent.document.visibilityState !== 'visible') stopCamera();
      }); } catch (_) {}
      window.addEventListener('pagehide', stopCamera);
      window.addEventListener('beforeunload', stopCamera);
    })();
    </script>""",
    height=0,
)

LANG = {
    "English": {"title":"🌿 CropGuard AI", "caption":"PlantVillage-trained MobileNetV2 • real inference • live CAM", "upload_title":"📷 Upload a leaf image", "upload_help":"Drag and drop a JPG, JPEG, PNG, or WEBP image into the box below, or tap Browse files.", "camera":"Or use your camera", "ready":"Ready to analyze", "analyze":"🔬 Analyze with trained model", "spinner":"Running TensorFlow inference and CAM…", "crop":"Crop", "diagnosis":"Diagnosis", "confidence":"Model confidence", "severity":"AI-derived severity estimate", "predictions":"Other model predictions", "advisory":"Treatment advisory", "history":"Actual scan history", "language":"🌐 Language"},
    "ಕನ್ನಡ": {"title":"🌿 CropGuard AI", "caption":"PlantVillage ತರಬೇತಿ ಪಡೆದ MobileNetV2 • ನೈಜ AI ವಿಶ್ಲೇಷಣೆ • CAM", "upload_title":"📷 ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ", "upload_help":"JPG, JPEG, PNG ಅಥವಾ WEBP ಚಿತ್ರವನ್ನು ಇಲ್ಲಿ ಹಾಕಿ ಅಥವಾ Browse files ಒತ್ತಿರಿ.", "camera":"ಅಥವಾ ಕ್ಯಾಮೆರಾ ಬಳಸಿ", "ready":"ವಿಶ್ಲೇಷಣೆಗೆ ಸಿದ್ಧ", "analyze":"🔬 ತರಬೇತಿ ಪಡೆದ ಮಾದರಿಯಿಂದ ವಿಶ್ಲೇಷಿಸಿ", "spinner":"TensorFlow ಮತ್ತು CAM ಮೂಲಕ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ…", "crop":"ಬೆಳೆ", "diagnosis":"ರೋಗನಿರ್ಣಯ", "confidence":"ಮಾದರಿ ವಿಶ್ವಾಸ", "severity":"AI ಅಂದಾಜಿನ ತೀವ್ರತೆ", "predictions":"ಇತರ ಮಾದರಿ ಮುನ್ಸೂಚನೆಗಳು", "advisory":"ಚಿಕಿತ್ಸಾ ಸಲಹೆ", "history":"ನಿಜವಾದ ಸ್ಕ್ಯಾನ್ ಇತಿಹಾಸ", "language":"🌐 ಭಾಷೆ"},
    "मराठी": {"title":"🌿 CropGuard AI", "caption":"PlantVillage प्रशिक्षित MobileNetV2 • वास्तविक AI विश्लेषण • CAM", "upload_title":"📷 पानाचा फोटो अपलोड करा", "upload_help":"JPG, JPEG, PNG किंवा WEBP फोटो येथे टाका किंवा Browse files दाबा.", "camera":"किंवा कॅमेरा वापरा", "ready":"विश्लेषणासाठी तयार", "analyze":"🔬 प्रशिक्षित मॉडेलने विश्लेषण करा", "spinner":"TensorFlow आणि CAM द्वारे विश्लेषण सुरू आहे…", "crop":"पीक", "diagnosis":"निदान", "confidence":"मॉडेलचा विश्वास", "severity":"AI अंदाजित तीव्रता", "predictions":"इतर मॉडेल अंदाज", "advisory":"उपचार सल्ला", "history":"वास्तविक स्कॅन इतिहास", "language":"🌐 भाषा"},
    "తెలుగు": {"title":"🌿 CropGuard AI", "caption":"PlantVillage శిక్షణ పొందిన MobileNetV2 • నిజమైన AI విశ్లేషణ • CAM", "upload_title":"📷 ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి", "upload_help":"JPG, JPEG, PNG లేదా WEBP చిత్రాన్ని ఇక్కడ డ్రాగ్ చేయండి లేదా Browse files నొక్కండి.", "camera":"లేదా కెమెరాను ఉపయోగించండి", "ready":"విశ్లేషణకు సిద్ధంగా ఉంది", "analyze":"🔬 శిక్షణ పొందిన మోడల్‌తో విశ్లేషించండి", "spinner":"TensorFlow మరియు CAM ద్వారా విశ్లేషిస్తోంది…", "crop":"పంట", "diagnosis":"రోగ నిర్ధారణ", "confidence":"మోడల్ విశ్వాసం", "severity":"AI అంచనా తీవ్రత", "predictions":"ఇతర మోడల్ అంచనాలు", "advisory":"చికిత్స సలహా", "history":"నిజమైన స్కాన్ చరిత్ర", "language":"🌐 భాష"},
}

if "site_language" not in st.session_state:
    st.session_state.site_language = "English"
    st.session_state.show_language_popup = True
else:
    st.session_state.show_language_popup = False

@st.dialog("🌐 Choose your language")
def language_popup():
    st.write("Select a language to use on CropGuard AI.")
    selected = st.selectbox(
        "Language",
        ["English", "ಕನ್ನಡ", "मराठी", "తెలుగు"],
        index=["English", "ಕನ್ನಡ", "मराठी", "తెలుగు"].index(st.session_state.site_language),
        key="language_popup_select",
    )
    if st.button("Continue", type="primary", use_container_width=True):
        st.session_state.site_language = selected
        st.session_state.show_language_popup = False
        st.rerun()

if st.session_state.get("show_language_popup", False):
    language_popup()

language = st.session_state.site_language
T = LANG[language]

with st.sidebar:
    configured_api = os.environ.get("CROPGUARD_API_URL", "http://localhost:8000").strip().rstrip("/")
    if configured_api and not configured_api.startswith(("http://", "https://")):
        configured_api = "https://" + configured_api
    api = st.text_input("Backend URL", configured_api).strip().rstrip("/")
    health = None
    try:
        health_response = requests.get(f"{api}/health", timeout=20)
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("model_loaded"):
            st.success("Real trained model loaded")
        else:
            st.warning("Model unavailable")
            st.caption(health.get("model_error", "No model loaded."))
    except (requests.RequestException, ValueError) as exc:
        st.error("Backend unavailable")
        st.caption(str(exc))

st.title(T["title"])
st.caption(T["caption"])

st.subheader(T["upload_title"])
st.caption(T["upload_help"])
upload = st.file_uploader("Drop your leaf image here", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False, key="leaf_upload")
st.caption(T["camera"])

# Keep the camera widget mounted only while actively taking a photo.
# After capture it is removed on the next rerun, which releases the browser camera.
if "camera_capture_bytes" not in st.session_state:
    st.session_state.camera_capture_bytes = None
    st.session_state.camera_capture_name = "camera.jpg"
    st.session_state.camera_capture_type = "image/jpeg"

camera = None
if st.session_state.camera_capture_bytes is None:
    camera = st.camera_input("Take a leaf photo", key="leaf_camera")
    if camera is not None:
        st.session_state.camera_capture_bytes = camera.getvalue()
        st.session_state.camera_capture_name = camera.name or "camera.jpg"
        st.session_state.camera_capture_type = camera.type or "image/jpeg"
        st.rerun()
else:
    camera = _CameraCapture(
        st.session_state.camera_capture_bytes,
        st.session_state.camera_capture_name,
        st.session_state.camera_capture_type,
    )
    if st.button("↻ Retake photo", key="retake_camera"):
        st.session_state.camera_capture_bytes = None
        st.rerun()

source = upload if upload is not None else camera

if source is not None:
    try:
        image_bytes = source.getvalue()
        image = Image.open(source).convert("RGB")
        st.success(f"Image received: {source.name or 'camera photo'}")
        st.image(image, caption="Your uploaded leaf image", width=500)
    except (UnidentifiedImageError, OSError):
        st.error("That file is not a valid image. Please choose a JPEG, PNG, WEBP, or camera image.")
        source = None
        image_bytes = None
else:
    image_bytes = None

if source is not None and image_bytes is not None:
    st.markdown(f"### {T['ready']}")
    if st.button(T["analyze"], type="primary", key="analyze_leaf"):
        if not api:
            st.error("Backend URL is missing.")
        else:
            try:
                with st.spinner(T["spinner"]):
                    response = requests.post(f"{api}/predict", files={"file": (source.name or "camera.jpg", image_bytes, source.type or "image/jpeg")}, timeout=180)
                if response.status_code >= 400:
                    try: details = response.json().get("detail", response.text)
                    except ValueError: details = response.text
                    st.error(f"Backend error ({response.status_code}): {details}")
                else:
                    data = response.json()
                    c1, c2, c3 = st.columns(3)
                    c1.metric(T["crop"], data["crop"]); c2.metric(T["diagnosis"], data["disease"]); c3.metric(T["confidence"], f"{data['confidence'] * 100:.2f}%")
                    st.metric(T["severity"], f"{data['severity_score']:.2f}%")
                    st.caption(f"Leaf area above activation threshold: {data['heatmap_coverage_percent']:.2f}% • Image SHA-256: {data['image_sha256']}")
                    st.image(data["heatmap_data_url"], caption="CAM derived from this uploaded image")
                    if data.get("storage_status") == "unavailable": st.warning("Prediction completed, but scan history could not be saved right now.")
                    st.subheader(T["predictions"])
                    st.table([{"class": p["label"], "probability": f"{p['probability'] * 100:.2f}%"} for p in data["top_predictions"]])
                    advisory = data["advisory"]
                    st.subheader(T["advisory"])
                    st.write(advisory["summary"])
                    for action in advisory["actions"]: st.write(f"• {action}")
                    for source_link in advisory.get("sources", []): st.caption(source_link)
            except requests.RequestException as exc: st.error(f"Backend/inference error: {exc}")

st.divider()
st.subheader("Model evidence")
try:
    response = requests.get(f"{api}/metrics", timeout=20)
    if response.ok:
        metrics = response.json(); st.write(f"Measured held-out test accuracy: **{metrics['test_accuracy'] * 100:.2f}%**")
    else: st.info("No measured metrics yet. Run the real training and evaluation pipeline first.")
except requests.RequestException: st.info("Start the FastAPI backend to view measured metrics.")

st.subheader(T["history"])
try:
    response = requests.get(f"{api}/history", timeout=20)
    if response.ok:
        history = response.json(); st.dataframe(history, use_container_width=True) if history else st.info("No scans yet.")
    else: st.info("Scan history is unavailable right now.")
except requests.RequestException: st.info("Start the FastAPI backend to view scan history.")

st.divider()
st.caption("CropGuard uses genuine model inference only; no prediction results or confidence scores are hardcoded.")
