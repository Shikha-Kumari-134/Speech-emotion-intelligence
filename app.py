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
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        filepath = tmp.name

    with st.spinner("🧠 Running Deep Neural Emotion Analysis..."):
        overall_emotion, confidence, all_preds = predict_emotion_with_confidence(filepath)        
        audio, sr = librosa.load(filepath, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)

    # 1. EMOTION DETECTION (CENTERED)
    st.markdown(f"""
        <div class="prediction-container">
            <p class="emotion-label">This audio sample shows you're feeling</p>
            <h1 class="emotion-value">{overall_emotion}</h1>
        </div>
    """, unsafe_allow_html=True)

    # 2. AUDIO PREVIEW
    a_left, a_mid, a_right = st.columns([1, 2, 1])
    with a_mid:
        st.audio(filepath)
        st.markdown(f"""
            <div style="text-align: center; margin-top: 15px;">
                <span class="stat-pill">LENGTH: {duration:.2f}s</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; margin:40px 0 20px 0;">
        <p style="font-size:1.05rem; color:#475569;">
            <b>Now, let’s take a closer look at the visual timeline of how emotions unfolded in your voice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Assuming you got these values from your new predict function
    emotion, confidence, all_preds = predict_emotion_with_confidence(filepath)

    conf_label = confidence_label(confidence)
    explanation = interpret_emotion(emotion, confidence)


    # 1. Display Confidence Score
    st.write(f"Confidence Score: {confidence:.2f} ({conf_label})")
    st.progress(confidence / 100) # Streamlit's built-in progress bar
    
    st.info(explanation)

    # 2. Calculate Intensity (RMS Energy)
    y, _ = librosa.load(filepath, sr=16000)
    rms = np.mean(librosa.feature.rms(y=y))
    intensity = min(100, rms * 500) # Scaling factor to make it look like a 0-100% bar
    st.write(f"**Vocal Intensity:** {intensity:.1f}%")
    st.progress(intensity / 100)

    # 3. Probability Distribution (The "Top 3" Chart)
    import pandas as pd
    label_encoder = joblib.load("data/label_encoder.pkl")

    chart_data = pd.DataFrame({
        'Emotion': label_encoder.classes_,
        'Probability': all_preds
    }).sort_values('Probability', ascending=False).head(3)

    st.write("#### Top 3 Probable Emotions")
    st.bar_chart(chart_data, x='Emotion', y='Probability')

    # 3. ADVANCED VISUALIZATION TABS
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📈 EMOTIONAL TIMELINE", "🔍 EXPLAINABLE AI (XAI)"])

    with tab1:
        st.markdown("#### Dynamic Emotional Evolution")
        st.write("Chunk-by-chunk analysis of emotional transitions.")
        
        # Segmented Logic
        chunk_duration = 1
        num_chunks = int(np.ceil(duration / chunk_duration))
        trend = []
        
        prog = st.progress(0)
        for i in range(num_chunks):
            start = int(i * chunk_duration * sr)
            end = int(min((i + 1) * chunk_duration * sr, len(audio)))
            chunk = audio[start:end]
            chunk_path = f"chunk_{i}.wav"
            sf.write(chunk_path, chunk, sr)
            
            # FIX: Unpack the tuple and only append the emotion string (the first item)
            emotion_label, _, _ = predict_emotion_with_confidence(chunk_path)
            trend.append(emotion_label) 
            
            os.remove(chunk_path)
            prog.progress((i + 1) / num_chunks)
        prog.empty()

        unique_emotions = list(dict.fromkeys(trend))
        numeric = [unique_emotions.index(e) for e in trend]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(numeric, marker='o', linewidth=4, color='#0f172a', markersize=10, markerfacecolor='#3b82f6')
        ax.set_yticks(range(len(unique_emotions)))
        ax.set_yticklabels([e.capitalize() for e in unique_emotions], fontweight='bold', color='#0f172a')
        ax.set_xlabel("Time (Seconds)", color='#64748b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('none')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Attention-Map (Gradient-weighted Activation)")
        st.write("Heatmap identifying the specific spectral regions driving the AI's decision.")
        
        mfcc_seq = extract_features(filepath)
        # Assuming model path is valid
        model = tf.keras.models.load_model("models/crnn_best_model.keras")
        input_tensor = tf.Variable(np.expand_dims(mfcc_seq, axis=0), dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(input_tensor)
            idx = tf.argmax(preds[0])
            score = preds[0][idx]

        grads = tape.gradient(score, input_tensor)[0]
        heatmap = tf.reduce_mean(grads, axis=-1).numpy()

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        librosa.display.specshow(mfcc_seq.T, x_axis="time", ax=ax2, cmap="magma")
        heatmap_2d = np.tile(heatmap, (mfcc_seq.shape[1], 1))
        img = ax2.imshow(heatmap_2d, origin="lower", aspect="auto", cmap="jet", alpha=0.35)
        
        ax2.set_title("Neural Attention Mapping", color='#0f172a', fontweight='bold')
        cbar = fig2.colorbar(img, ax=ax2, pad=0.02)
        cbar.set_label('Attention Intensity', color="#0f172a")
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists(filepath):
        os.remove(filepath)

else:
    st.markdown("""
        <div style="text-align: center; padding: 100px; color: #94a3b8 !important; letter-spacing: 2px;">
            SYSTEM AWAITING UPLINK...
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; padding: 60px 0; font-size: 0.8rem; letter-spacing: 2px; opacity: 0.5;">
        VibeCheck.AI // SPEECH EMOTION // 2025
    </div>
""", unsafe_allow_html=True)
