import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import soundfile as sf
import base64
from src.predict import predict_emotion_with_confidence, extract_features
import joblib
import pandas as pd
import tempfile

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="VibeCheck.AI",
    page_icon="🎧",
    layout="wide",
)
BASE_DIR = os.path.dirname(__file__)
ENCODER_PATH = os.path.join(BASE_DIR, "data", "label_encoder.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crnn_best_model.keras")
label_encoder = joblib.load(ENCODER_PATH)

@st.cache_resource
def load_vibe_model():
    return tf.keras.models.load_model(MODEL_PATH)
    
model = load_vibe_model()        

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Path to your local image file
image_file = "src/background.avif" 
try:
    base64_image = get_base64_image(image_file)
    bg_style = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.92)),
                    url("data:image/jpeg;base64,{base64_image}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    </style>"""
except:
    bg_style = "<style>.stApp { background-color: #f8fafc; }</style>"

# =================================================
# THE "BILLION DOLLAR" CINEMATIC CSS
# =================================================
st.markdown(bg_style, unsafe_allow_html=True)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&family=Space+Grotesk:wght@300;500;700&display=swap');
    h1, h2, h3, h4, p, span, div { color: #0f172a !important; font-family: 'Inter', sans-serif; }
    /* Fix the shaking on the right side */
    .stApp {
        overflow-x: hidden !important;
    }
    
    /* Force vertical scrollbar to always stay put */
    html {
        overflow-y: scroll;
    }
    
    /* Remove default padding that causes layout shifts */
    .block-container {
        padding-top: 1rem !important;
        max-width: 1100px !important;
    }
    /* HERO SECTION */
    .hero-container { min-height:450px; overflow:hidden; padding: 80px 0 40px 0; text-align: center; animation: fadeIn 1.5s ease-in;
    margin: 0 auto 2rem auto !important;}
    .main-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem;
        letter-spacing: 8px; text-transform: uppercase; font-weight: 500;
        margin-bottom: 2rem; color: #3b82f6 !important;
    }
    .hero-text {
        font-family: 'Space Grotesk', sans-serif; font-size: 5.5rem;
        font-weight: 700; line-height: 1.1; letter-spacing: -3px; margin-bottom: 2rem;
    }
    .sub-hero {
        display: block !important;
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        max-width: 650px !important;
        padding: 0 20px !important;  /* Perfect side padding */
        line-height: 1.6 !important;
        animation: fadeIn 2s ease-in;
    }
    /* CENTERED PREDICTION SECTION */
    .prediction-container { text-align: center; padding: 40px 0; animation: fadeIn 1s ease-out; }
    .emotion-label { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 400; opacity: 0.8; }
    .emotion-value {
        font-family: 'Space Grotesk', sans-serif; font-size: 8rem;
        font-weight: 700; letter-spacing: -6px; text-transform: uppercase;
        line-height: 1; margin-top: 10px;
    }
    /* CARDS */
    .custom-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 30px;
        margin-top: 20px;
        backdrop-filter: blur(10px);
    }
    /* CUSTOM UPLOAD CARD */
    .custom-upload-card {
        background: linear-gradient(145deg, #020617, #0f172a);
        border: 2px dashed #3b82f6;
        border-radius: 26px;
        padding: 45px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .custom-upload-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 50px rgba(59,130,246,0.25);
    }
    .custom-upload-card h3 {
        color: #e5e7eb !important;
        margin-top: 15px;
    }
    .custom-upload-card p {
        color: #94a3b8 !important;
    }
    .upload-icon {
        font-size: 3rem;
    }
        
    .stat-pill {
        background: #0f172a; color: #ffffff !important; padding: 8px 20px;
        border-radius: 100px; font-size: 0.9rem; font-weight: 600;
        display: inline-block; margin: 5px;
    }
    /* DESIGNER BUTTON STYLING */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        width: 100% !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6) !important;
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(1px) !important;
    }
    
    /* =========================
   STREAMLIT FILE UPLOADER FIX
   ========================= */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.95) !important; /* dark slate */
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    /* Drag & drop text */
    [data-testid="stFileUploader"] div {
        color: #e5e7eb !important; /* light gray */
        font-weight: 500;
    }
    /* "Drag and drop file here" */
    [data-testid="stFileUploader"] small {
        color: #cbd5f5 !important;
        font-size: 0.9rem;
    }
    /* Limit text */
    [data-testid="stFileUploader"] span {
        color: #94a3b8 !important;
    }
    /* Browse files button */
    [data-testid="stFileUploader"] button {
        background: #0f172a !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #1e293b !important;
    }
        /* Fix plot and tab containers to stop jumping */
    [data-testid="stVerticalBlock"] > div {
        transition: none !important; /* Disables the sliding animation that looks like shaking */
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        min-height: 400px; /* Reserve space so the page doesn't collapse while loading plots */
    }
    /* TABS CUSTOMIZATION */
    .stTabs [data-baseweb="tab-list"] { justify-content: center; gap: 40px; }
    .stTabs [data-baseweb="tab"] { font-weight: 700; color: #64748b !important; }
    .stTabs [aria-selected="true"] { color: #0f172a !important; border-bottom-color: #3b82f6 !important; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =================================================
# PAGE 1: HERO & INGESTION
# =================================================
st.markdown(f"""
<div class="hero-container">
    <div class="main-title">[ Speech Emotion Intelligence ]</div>
    <div class="hero-text">Listen To The<br>UNSAID <br>In Your Voice</div>
    <div class="sub-hero">
        This AI powered Speech Analyzer deciphers human emotions in real-time. 
        The most advanced way to monitor sentiment and understand vocal patterns.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 12px;">
    <h3>🎧 Let's first upload an Audio File</h3>
    <p style="color:#475569; font-size:1rem;">
        Ensure the audio is clear and between 3–30 seconds for maximum extraction accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Audio 🎵", 
    type=['wav', 'mp3', 'm4a', 'mp4', 'ogg', 'flac', 'aac'],  # ALL FORMATS
    help="Supports WAV, MP3, M4A, MP4, OGG, FLAC, AAC"
)

st.markdown("---")

def confidence_label(confidence):
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    else:
        return "Low"

def interpret_emotion(emotion, confidence):
    if emotion.lower() == "angry":
        return (
            "This prediction is influenced by higher vocal energy and sharper pitch variations. "
            "Such patterns are commonly associated with anger-like expressions."
        )

    if emotion.lower() == "neutral":
        return (
            "The voice shows balanced pitch and energy levels, which usually indicate a neutral tone."
        )

    return (
        "The emotion is inferred from a combination of pitch, energy, and spectral features "
        "learned during training."
    )

# =================================================
# PAGE 2: ANALYSIS
# =================================================
if uploaded_file is not None:
    # 1. Safe temp paths
    processed_path = "temp.wav"
    
    # 2. Save uploaded file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        filepath = tmp.name

    try:
        with st.spinner("🧠 Decoding & Analyzing Audio..."):
            # LIBROSA handles MP3/MP4/M4A natively via ffmpeg (ALREADY INSTALLED)
            audio, sr = librosa.load(filepath, sr=16000, mono=True, duration=30.0)
            
            # Export WAV for model (predict.py expects WAV)
            sf.write(processed_path, audio, sr)
            
            # Predict
            overall_emotion, confidence, all_preds = predict_emotion_with_confidence(processed_path)
            duration = librosa.get_duration(y=audio, sr=sr)

        # Your UI (UNCHANGED - perfect)
        if overall_emotion:
            st.markdown(f"""
                <div class="prediction-container">
                    <p class="emotion-label">This audio sample shows you're feeling</p>
                    <div class="emotion-value">{overall_emotion}</div>
                </div>
            """, unsafe_allow_html=True)

            # Play ORIGINAL file (MP3 works!)
            a_left, a_mid, a_right = st.columns([1, 2, 1])
            with a_mid:
                st.audio(uploaded_file, format='audio/wav')  # Native playback
                st.markdown(f"""
                    <div style="text-align: center; margin-top: 8px;">
                        <span class="stat-pill">LENGTH: {duration:.2f}s</span>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="text-align:center; margin:40px 0 20px 0;"><b>Now, let’s take a closer look at the visual timeline.</div>', unsafe_allow_html=True)

        # Confidence and Explanation
        conf_label = confidence_label(confidence)
        st.write(f"Confidence Score: {confidence:.2f} ({conf_label})")
        st.progress(float(confidence / 100))
        st.info(interpret_emotion(overall_emotion, confidence))

        # Intensity
        rms = np.mean(librosa.feature.rms(y=audio))
        intensity = min(100, rms * 500)
        st.write(f"**Vocal Intensity:** {intensity:.1f}%")
        st.progress(float(intensity / 100))

        # Probability Chart
        chart_data = pd.DataFrame({
            'Emotion': label_encoder.classes_,
            'Probability': all_preds
        }).sort_values('Probability', ascending=False).head(3)
        st.write("#### Top 3 Probable Emotions")
        st.bar_chart(chart_data, x='Emotion', y='Probability')

        # TABS
        tab1, tab2 = st.tabs(["📈 EMOTIONAL TIMELINE", "🔍 EXPLAINABLE AI (XAI)"])

        with tab1:
            st.markdown("#### Dynamic Emotional Evolution")
            chunk_duration = 1
            num_chunks = int(np.ceil(duration / chunk_duration))
            trend = []
            progress_placeholder = st.empty() 
            
            for i in range(num_chunks):
                start = int(i * sr)
                end = int(min((i + 1) * sr, len(audio)))
                chunk_file = f"chunk_{i}.wav"
                sf.write(chunk_file, audio[start:end], sr)
                e_label, _, _ = predict_emotion_with_confidence(chunk_file)
                trend.append(e_label) 
                os.remove(chunk_file)
                progress_placeholder.progress(float((i + 1) / num_chunks))
            progress_placeholder.empty()

            unique_emotions = list(dict.fromkeys(trend))
            numeric = [unique_emotions.index(e) for e in trend]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(numeric, marker='o', linewidth=4, color='#0f172a', markerfacecolor='#3b82f6')
            ax.set_yticks(range(len(unique_emotions)))
            ax.set_yticklabels([e.capitalize() for e in unique_emotions], fontweight='bold')
            st.pyplot(fig, use_container_width=True)

        with tab2:
            st.markdown("#### Attention-Map (Gradient-weighted Activation)")
            if st.button("Generate Neural Attention Map", use_container_width=True):
                with st.spinner("🧠 Scanning..."):
                    mfcc_seq = extract_features(processed_path)
                    input_tensor = tf.Variable(np.expand_dims(mfcc_seq, axis=0), dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        preds = model(input_tensor)
                        score = preds[0][tf.argmax(preds[0])]
                    grads = tape.gradient(score, input_tensor)[0]
                    heatmap = tf.reduce_mean(grads, axis=-1).numpy()
                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                    librosa.display.specshow(mfcc_seq.T, x_axis="time", ax=ax2, cmap="magma")
                    ax2.imshow(np.tile(heatmap, (mfcc_seq.shape[1], 1)), origin="lower", aspect="auto", cmap="jet", alpha=0.35)
                    st.pyplot(fig2, use_container_width=True)

    except Exception as ex:
        st.error(f"Audio failed: {ex}")
    finally:
        # Clean up
        for path in [filepath, processed_path]:
            if os.path.exists(path):
                os.remove(path)

else:
    st.markdown('<div style="text-align: center; padding: 100px; color: #94a3b8; letter-spacing: 2px;">SYSTEM AWAITING UPLINK...</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; padding: 60px 0; font-size: 0.8rem; letter-spacing: 2px; opacity: 0.5;">VibeCheck.AI // 2025</div>', unsafe_allow_html=True)
