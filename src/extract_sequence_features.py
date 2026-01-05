import os 
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import joblib

METADATA = "data/metadata.csv"
OUT_DIR="data"
os.makedirs(OUT_DIR, exist_ok=True)

#Feature params
N_MFCC = 40
SAMPLE_RATE = 16000
PERCENTILE_FOR_MAXLEN= 95

#Load metadata
df= pd.read_csv(METADATA)
#filter unknown emotions if any
df=df[df['emotion']. notnull()]. reset_index(drop=True)

print("Total samples in  metadata: ", len(df))

#function to get mfcc sequence (time_steps , n_mfcc)
def mfcc_sequence(path, n_mfcc=N_MFCC, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr)
    if y is None or len(y) == 0:
        raise ValueError("Empty audio")
    
    # 1. Standard MFCCs (The "Photo")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 2. Delta MFCCs (The "Speed/Velocity")
    delta_mfcc = librosa.feature.delta(mfcc)
    
    # 3. Delta-Delta MFCCs (The "Acceleration")
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Stack them vertically to create 120 features per frame
    features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
    
    if features.shape[1] == 0:
        raise ValueError("Feature extraction resulted in zero frames")
        
    return features.T # Returns (time_steps, 120)

#Scan to collect lengths
lengths=[]
print("Scanning files to compute MFCC frame lengths...")
for p in tqdm(df['file_path'].values, total=len(df)):
    try:
        seq=mfcc_sequence(p)
        lengths.append(seq.shape[0])
    except Exception as e:
        lengths.append(0)

lengths = np.array(lengths)
valid_lengths = lengths[lengths > 0 ]
if len(valid_lengths)==0:
    raise RuntimeError("No valid audio files found. Check librosa/ffmpeg and file formats.")

#Choose MAX_LEN using percentile
MAX_LEN=int(np.percentile(valid_lengths, PERCENTILE_FOR_MAXLEN))
MAX_LEN=max(50, MAX_LEN)
print("Chosen MAX_LEN (frames):", MAX_LEN)


#Build label encoder
emotions = sorted(df['emotion'].unique())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df["emotion"])   # train on actual emotion names
joblib.dump(le, "data/label_encoder.pkl")

print("Saved sklearn LabelEncoder with classes:", le.classes_)

#Extract and pad/truncate
def pad_truncate(seq,max_len):
    if seq.shape[0]<max_len:
        pad_width = max_len - seq.shape[0]
        return np.pad(seq, ((0,pad_width), (0,0)),mode='constant', constant_values=0.0)
    else:
        return seq[:max_len,:]

X_list = []
y_list = []
skipped = 0
errors = []

print("Extracting, padding and saving MFCC sequences...")
for i, row in tqdm(df.iterrows(), total =len(df)):
    p=row['file_path']
    emo=row['emotion']

    try:
        seq=mfcc_sequence(p)
        seq=pad_truncate(seq, MAX_LEN)
        X_list.append(seq.astype(np.float32))
        y_list.append(le.transform([emo])[0])
    except Exception as e:
        skipped+=1
        errors.append((p, str(e)))

#Convert to arrays
if len(X_list) ==0:
    print("ERROR: no sequence extracted. See sample errors below:")
    for i, (p,msg) in enumerate(errors[:10]):
        print(i+1, p, "->", msg)
    raise RuntimeError("No MFCC sequence created. Fix issues above (librosa, file formats).")

X=np.stack(X_list, axis=0)
y=np.array(y_list, dtype=np.int32)

print("Built X Shape:",X.shape, "y shape:", y.shape," skipped:",skipped)

#Save
np.save(os.path.join(OUT_DIR, "X_seq.npy"),X)
np.save(os.path.join(OUT_DIR, "y.npy"),y)

#Save metadata about sequences
meta={"n_mfcc": N_MFCC, "max_len": MAX_LEN, "emotion_classes": le.classes_.tolist()}
with open(os.path.join(OUT_DIR,"meta_seq.json"),"w") as f:
    json.dump(meta, f, indent=2)

print("Saved X_seq.npy , y.npy, meta_seq.json in",OUT_DIR)

if skipped>0:
    print("Some files were skipped. First 10 errors:")
    for i, (p,msg) in enumerate(errors[:10]):
        print(i+1 , p, "->", msg)