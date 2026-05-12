import librosa
import numpy as np
import joblib
import pandas as pd


model = joblib.load('drop_model.pkl')
scaler = joblib.load('scaler.pkl')


def find_epic_moment(file_path) -> None:
    print(f"Analyzing the track: {file_path}...")
    y_audio, sr = librosa.load(file_path)
    rms = librosa.feature.rms(y=y_audio)[0]
    cent = librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_audio)[0]

    X_new = []
    timestamps = [] 

    window_size = 86
    step = 43

    for start in range(0, len(rms) - window_size, step):
        w_rms = rms[start : start + window_size]
        w_cent = cent[start : start + window_size]
        w_zcr = zcr[start : start + window_size]
        
        row = [np.mean(w_rms), np.std(w_rms), np.mean(w_cent), np.std(w_cent), np.mean(w_zcr)]
        X_new.append(row)
        timestamps.append(start / 43)

    columns = ['mean_rms', 'std_rms', 'mean_cent', 'std_cent', 'mean_zcr']
    X_new_df = pd.DataFrame(X_new, columns=columns)
    X_scaled = scaler.transform(X_new_df)
    probs = model.predict_proba(X_scaled)[:, 1]

    best_idx = np.argmax(probs)
    epic_time = timestamps[best_idx]
    
    minutes = int(epic_time // 60)
    seconds = int(epic_time % 60)

    print(f"\n🔥 THE MOST EPIC MOMENT FOUND!")
    print(f"Timecode: {minutes:02d}:{seconds:02d}")
    print(f"Model Confidence: {probs[best_idx]*100:.1f}%")

    return None