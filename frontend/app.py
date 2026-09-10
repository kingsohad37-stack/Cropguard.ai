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
:root { --cg-ink:#163222; --cg-muted:#52675a; --cg-panel:rgba(255,255,255,.68); --cg-line:rgba(255,255,255,.72); --cg-shadow:0 18px 55px rgba(19,55,31,.14); }
html,body,.stApp { font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Inter",sans-serif!important; color:var(--cg-ink)!important; }
body { background:#dcebdc!important; }
[data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"]>.main, .main, .main .block-container { background:transparent!important; }
[data-testid="stAppViewContainer"] { min-height:100vh!important; background-image:linear-gradient(180deg,rgba(245,250,246,.18),rgba(224,239,226,.38)),url('/app/static/cropguard-bg.jpg')!important; background-size:cover!important; background-position:center center!important; background-repeat:no-repeat!important; background-attachment:fixed!important; }
[data-testid="stAppViewContainer"]:before { content:""!important; position:fixed!important; inset:0!important; z-index:0!important; pointer-events:none!important; background:radial-gradient(circle at 12% 8%,rgba(255,255,255,.70),transparent 28%),radial-gradient(circle at 88% 88%,rgba(100,180,117,.15),transparent 34%)!important; }
[data-testid="stHeader"] { background:transparent!important; }
.main .block-container { position:relative!important; z-index:1!important; width:min(100%,1120px)!important; max-width:1120px!important; padding:2.25rem 1.25rem 4rem!important; }
h1,h2,h3,h4 { color:var(--cg-ink)!important; font-weight:800!important; letter-spacing:-.035em!important; line-height:1.08!important; }
h1 { font-size:clamp(2.45rem,6vw,4.25rem)!important; margin-bottom:.35rem!important; }
p,[data-testid="stCaptionContainer"],label { color:var(--cg-muted)!important; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--cg-panel)!important; -webkit-backdrop-filter:blur(20px) saturate(125%)!important; backdrop-filter:blur(20px) saturate(125%)!important; border:1px solid var(--cg-line)!important; border-radius:24px!important; box-shadow:var(--cg-shadow)!important; }
[data-testid="stMetric"] { background:rgba(255,255,255,.78)!important; border:1px solid rgba(255,255,255,.84)!important; border-radius:20px!important; box-shadow:0 12px 36px rgba(19,55,31,.10)!important; }
[data-testid="stMetricValue"],[data-testid="stMetricLabel"] { color:var(--cg-ink)!important; }
[data-testid="stFileUploaderDropzone"],[data-testid="stCameraInput"] { background:rgba(255,255,255,.76)!important; border:1px solid rgba(255,255,255,.90)!important; border-radius:20px!important; box-shadow:0 12px 36px rgba(19,55,31,.09)!important; }
[data-testid="stFileUploaderDropzone"] * { color:var(--cg-ink)!important; }
.stTextInput input,[data-baseweb="select"]>div { background:rgba(255,255,255,.90)!important; color:var(--cg-ink)!important; border:1px solid rgba(30,82,48,.14)!important; border-radius:16px!important; min-height:2.9rem!important; }
[data-baseweb="select"] * { color:var(--cg-ink)!important; }
.stButton>button,.stDownloadButton>button { border-radius:999px!important; min-height:2.9rem!important; padding:.62rem 1.18rem!important; border:1px solid rgba(255,255,255,.88)!important; color:#173a24!important; background:linear-gradient(145deg,rgba(255,255,255,.94),rgba(235,246,237,.82))!important; box-shadow:0 9px 26px rgba(24,79,42,.12),inset 0 1px 0 rgba(255,255,255,1)!important; font-weight:750!important; transition:transform .2s cubic-bezier(.4,0,.2,1),box-shadow .2s ease!important; }
.stButton>button:hover,.stDownloadButton>button:hover { transform:scale(1.03)!important; box-shadow:0 14px 34px rgba(24,79,42,.18),inset 0 1px 0 rgba(255,255,255,1)!important; }
.stButton>button:active,.stDownloadButton>button:active { transform:scale(.97)!important; }
.stButton>button[kind="primary"] { color:#fff!important; background:linear-gradient(135deg,#2c8b50,#17683b)!important; box-shadow:0 12px 32px rgba(25,111,59,.28),inset 0 1px 0 rgba(255,255,255,.32)!important; }
[data-testid="stDialog"],[data-testid="stDialog"]>div,[role="dialog"] { color-scheme:light!important; }
[data-testid="stDialog"]>div,[role="dialog"] { width:min(92vw,520px)!important; max-width:calc(100vw - 24px)!important; background:rgba(247,251,247,.98)!important; color:var(--cg-ink)!important; border:1px solid rgba(255,255,255,.98)!important; border-radius:26px!important; box-shadow:0 28px 80px rgba(15,46,26,.24)!important; overflow:hidden!important; }
[data-testid="stDialog"] h1,[data-testid="stDialog"] h2,[data-testid="stDialog"] h3,[data-testid="stDialog"] p,[data-testid="stDialog"] label,[role="dialog"] h1,[role="dialog"] h2,[role="dialog"] h3,[role="dialog"] p,[role="dialog"] label { color:var(--cg-ink)!important; }
[data-testid="stDialog"] [data-baseweb="select"]>div,[role="dialog"] [data-baseweb="select"]>div { background:#fff!important; color:var(--cg-ink)!important; border:1px solid rgba(31,93,51,.18)!important; border-radius:16px!important; }
[data-testid="stDialog"] [data-baseweb="select"] *,[role="dialog"] [data-baseweb="select"] * { color:var(--cg-ink)!important; }
[data-testid="stDialog"] .stButton>button,[role="dialog"] .stButton>button { width:100%!important; }
[data-testid="stImage"] img { display:block!important; width:100%!important; max-width:100%!important; height:auto!important; object-fit:contain!important; border-radius:20px!important; border:1px solid rgba(255,255,255,.82)!important; box-shadow:0 14px 42px rgba(19,55,31,.13)!important; }
[data-testid="stAlert"],[data-testid="stDataFrame"] { border-radius:18px!important; }
hr { border:0!important; border-top:1px solid rgba(35,79,47,.13)!important; }
@media(max-width:700px){ [data-testid="stAppViewContainer"]{background-attachment:scroll!important;background-position:center top!important;} .main .block-container{padding:1.15rem .72rem 2.5rem!important;} h1{font-size:2.55rem!important;} h2{font-size:1.55rem!important;} h3{font-size:1.22rem!important;} [data-testid="stVerticalBlockBorderWrapper"]{border-radius:20px!important;} [data-testid="stDialog"]>div,[role="dialog"]{border-radius:22px!important;} }
</style>\n<style>\n/* Final stable CropGuard visual pass */\nhtml,body,.stApp,.stMarkdown,p,li,label,[data-testid="stCaptionContainer"]{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Inter",sans-serif!important;}\n.main .block-container{width:min(100% - 32px,1080px)!important;max-width:1080px!important;margin:0 auto!important;}\n.main .block-container p,.main .block-container li{font-size:1.08rem!important;line-height:1.62!important;font-weight:600!important;color:#294637!important;}\n.main .block-container h1{font-size:clamp(2.7rem,7vw,4.4rem)!important;font-weight:850!important;line-height:1.02!important;}\n.main .block-container h2{font-size:clamp(2rem,5vw,3rem)!important;font-weight:850!important;line-height:1.08!important;}\n.main .block-container h3{font-size:clamp(1.35rem,3.5vw,1.9rem)!important;font-weight:800!important;}\n.main .block-container [data-testid="stVerticalBlockBorderWrapper"]{width:100%!important;box-sizing:border-box!important;padding:clamp(18px,3vw,30px)!important;}\n.main .block-container [data-testid="column"]{min-width:0!important;width:100%!important;flex:1 1 100%!important;}\n.main .block-container [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:1rem!important;}\n[data-testid="stDataFrame"],[data-testid="stTable"]{width:100%!important;max-width:100%!important;overflow:hidden!important;}\n/* Debug/code-looking output is not part of the user-facing design. */\npre,[data-testid="stCodeBlock"],.stCodeBlock{display:none!important;}\n@media(max-width:700px){\n .main .block-container{width:calc(100% - 20px)!important;padding:1rem 0 3rem!important;}\n .main .block-container p,.main .block-container li{font-size:1.02rem!important;line-height:1.55!important;}\n .main .block-container h1{font-size:2.55rem!important;}\n .main .block-container h2{font-size:1.9rem!important;}\n .main .block-container h3{font-size:1.28rem!important;}\n .main .block-container [data-testid="stVerticalBlockBorderWrapper"]{padding:18px!important;border-radius:22px!important;}\n}\n</style>\n
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
