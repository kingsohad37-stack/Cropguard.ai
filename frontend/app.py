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

    body {
        background: #eaf3ec url('/app/static/cropguard-bg.jpg') center / cover fixed no-repeat !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(135deg, rgba(245,250,246,.74), rgba(226,240,230,.82)),
            url('/app/static/cropguard-bg.jpg') center / cover fixed no-repeat !important;
        min-height: 100vh;
    }

    [data-testid="stAppViewContainer"]::before {
        content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background: radial-gradient(circle at 15% 5%, rgba(255,255,255,.68), transparent 36%);
    }

    .main .block-container { position: relative; z-index: 1; }

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
        background: rgba(246,250,247,.68) !important; backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
        border-right: 1px solid rgba(48,92,61,.10) !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.42) !important; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
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
        transition: all .2s ease !important;
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
