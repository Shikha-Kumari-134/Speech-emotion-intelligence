import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import soundfile as sf
import base64
from src.predict import predict_emotion_with_confidence, extract_features
import joblib
import tempfile
from pydub import AudioSegment

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/cnn_best_model.keras")

@st.cache_resource
def load_label_encoder():
    return joblib.load("data/label_encoder.pkl")

model = load_model()
label_encoder = load_label_encoder()

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="VibeCheck.AI",
    page_icon="🎧",
    layout="wide",
)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

image_file = "background.avif"
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

    /* HERO SECTION */
    .hero-container { padding: 80px 0 40px 0; text-align: center; animation: fadeIn 1.5s ease-in; }
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
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 400;
        line-height: 1.6;
        color: #475569 !important; /* Elegant Slate Blue-Grey */
        max-width: 700px;
        margin: 0 auto 3rem auto; /* Centers the block and adds bottom spacing */
        opacity: 0.9;
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

    /* TABS CUSTOMIZATION */
    .stTabs [data-baseweb="tab-list"] { justify-content: center; gap: 40px; }
    .stTabs [data-baseweb="tab"] { font-weight: 700; color: #64748b !important; }
    .stTabs [aria-selected="true"] { color: #0f172a !important; border-bottom-color: #3b82f6 !important; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =================================================
# HERO SECTION
# =================================================
st.markdown("""
<div class="hero-container">
    <div class="main-title">[ Speech Emotion Intelligence ]</div>
    <div class="hero-text">Listen To The<br>UNSAID <br>In Your Voice</div>
    <p class="sub-hero">
        This AI powered Speech Analyzer deciphers human emotions in real-time.
        The most advanced way to monitor sentiment and understand vocal patterns.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 12px;">
    <h3>🎧 Let's first provide an Audio Input</h3>
    <p style="color:#475569;">
        Ensure the audio is clear and between 3–30 seconds.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Audio file",
    type=["wav", "mp3", "m4a", "ogg", "mp4"]
)

st.markdown("---")

def confidence_label(conf):
    return "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low"

def interpret_emotion(emotion, confidence):
    if emotion.lower() == "angry":
        return "Higher vocal energy and sharp pitch variations suggest anger."
    if emotion.lower() == "neutral":
        return "Balanced pitch and energy indicate a neutral tone."
    return "Emotion inferred from pitch, energy, and spectral features."

# =================================================
# ANALYSIS
# =================================================
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio = AudioSegment.from_file(uploaded_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(tmp.name, format="wav")
        filepath = tmp.name

    with st.spinner("🧠 Running Deep Neural Emotion Analysis..."):
        emotion, confidence, all_preds = predict_emotion_with_confidence(filepath)
        y, sr = librosa.load(filepath, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)

    st.markdown(f"""
        <div class="prediction-container">
            <p class="emotion-label">This audio sample shows you're feeling</p>
            <h1 class="emotion-value">{emotion}</h1>
        </div>
    """, unsafe_allow_html=True)

    st.audio(filepath)
    st.write(f"**Confidence:** {confidence:.2f} ({confidence_label(confidence)})")
    st.progress(confidence)

    st.info(interpret_emotion(emotion, confidence))

    chart_data = pd.DataFrame({
        "Emotion": label_encoder.classes_,
        "Probability": all_preds
    }).sort_values("Probability", ascending=False).head(3)

    st.subheader("Top 3 Probable Emotions")
    st.bar_chart(chart_data, x="Emotion", y="Probability")

    # ================= EMOTIONAL TIMELINE =================
    st.subheader("📈 Emotional Timeline")
    chunk_duration = 1
    num_chunks = int(np.ceil(duration / chunk_duration))
    trend = []

    prog = st.progress(0.0)
    for i in range(num_chunks):
        start = int(i * chunk_duration * sr)
        end = int(min((i + 1) * chunk_duration * sr, len(y)))
        chunk = y[start:end]

        chunk_path = f"chunk_{i}.wav"
        sf.write(chunk_path, chunk, sr)

        e, _, _ = predict_emotion_with_confidence(chunk_path)
        trend.append(e)

        os.remove(chunk_path)
        prog.progress((i + 1) / num_chunks)

    prog.empty()

    unique = list(dict.fromkeys(trend))
    numeric = [unique.index(e) for e in trend]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(numeric, marker="o", linewidth=3)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique)
    ax.set_xlabel("Time (Seconds)")
    st.pyplot(fig)

    # ================= XAI =================
    st.subheader("🔍 Explainable AI (XAI)")
    mfcc_seq = extract_features(filepath)
    input_tensor = tf.Variable(np.expand_dims(mfcc_seq, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = model(input_tensor)
        idx = tf.argmax(preds[0])
        score = preds[0][idx]

    grads = tape.gradient(score, input_tensor)[0]
    heatmap = tf.reduce_mean(grads, axis=-1).numpy()

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    librosa.display.specshow(mfcc_seq.T, x_axis="time", ax=ax2, cmap="magma")
    ax2.imshow(np.tile(heatmap, (mfcc_seq.shape[1], 1)), alpha=0.35, aspect="auto")
    st.pyplot(fig2)

    os.remove(filepath)

else:
    st.markdown("""
        <div style="text-align: center; padding: 100px; color: #94a3b8;">
            SYSTEM AWAITING UPLINK...
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:40px; opacity:0.5;">
    VibeCheck.AI // 2025
</div>
""", unsafe_allow_html=True)
