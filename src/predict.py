import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# Load trained CRNN model
model = load_model("models/crnn_best_model.keras")

# Load the SAME label encoder used during extraction + training
label_encoder = joblib.load("data/label_encoder.pkl")

def extract_features(file_path, n_mfcc=40, max_frames=142):
    y, sr = librosa.load(file_path, sr=16000)
    
    # 1. Base MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # FIX: Calculate a safe width for delta calculation
    n_frames = mfcc.shape[1]
    
    # librosa.feature.delta needs width < n_frames and width must be odd
    if n_frames < 3:
        # If the audio is too short to even get 3 frames, fill deltas with zeros
        delta_mfcc = np.zeros_like(mfcc)
        delta2_mfcc = np.zeros_like(mfcc)
    else:
        # Use a width of 9, or the largest odd number smaller than n_frames
        safe_width = min(9, n_frames if n_frames % 2 != 0 else n_frames - 1)
        delta_mfcc = librosa.feature.delta(mfcc, width=safe_width)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=safe_width)
    
    # Stack them to get 120 features (40 MFCC + 40 Delta + 40 Delta2)
    features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)

    # Pad/trim logic remains the same
    if features.shape[1] < max_frames:
        features = np.pad(features, ((0, 0), (0, max_frames - features.shape[1])), mode="constant")
    else:
        features = features[:, :max_frames]

    return features.T.astype(np.float32)

def predict_emotion_with_confidence(file_path):
    x = extract_features(file_path)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0] # Get the first (and only) prediction array

    class_idx = np.argmax(preds)
    confidence = preds[class_idx] * 100 # Convert to percentage
    emotion = label_encoder.inverse_transform([class_idx])[0]

    return emotion, confidence, preds # Also return all preds for a bar chart