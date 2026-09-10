"""Streamlit client for genuine CropGuard inference only."""
import os
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="CropGuard AI", page_icon="🌿", layout="wide")

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
camera = st.camera_input("Take a leaf photo", key="leaf_camera")
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
