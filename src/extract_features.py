from pathlib import Path
import librosa
import numpy as np


def song_processing() -> tuple[list[list[float]], list[int]]:
    dataset_dir = Path("audio_dataset")

    data_set_x = []
    data_set_y = []

    drops_info = {
        "Kordhell - Murder In My Mind": (20, 50),
        "SHADXWBXRN - DAMAGE": (15, 45),
        "Skrillex - Bangarang feat. Sirah [Official Music Video]": (26, 55),
        "KSLV - Disaster": (12, 40),
        "Hensonn - Sahara": (25, 60),
        "Ghostface Playa - Why Not": (18, 50),
        "DVRST - Close Eyes": (22, 55),
        "INTERWORLD - METAMORPHOSIS": (28, 65),
        "MoonDeity - NEON BLADE": (35, 80),
        "Aero Chord - Surface": (45, 90),
        "DJ Snake, Lil Jon - Turn Down for What": (18, 48),
        "Martin Garrix - Animals (Official Video)": (65, 110)
    }

    for audio_file in dataset_dir.glob("*.wav"):

        track_name = audio_file.stem
        y_audio, sr = librosa.load(audio_file)
        rms = librosa.feature.rms(y=y_audio)[0]
        cent = librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y_audio)[0]

        window_size = 86  
        step = 43

        total_frames = len(rms)

        for start in range(0, total_frames - window_size, step):
            window_rms = rms[start : start + window_size]
            window_cent = cent[start : start + window_size]
            window_zcr = zcr[start : start + window_size]
                
            mean_rms = np.mean(window_rms)
            std_rms = np.std(window_rms)
            mean_cent = np.mean(window_cent)
            std_cent = np.std(window_cent)
            mean_zcr = np.mean(window_zcr)

            current_sec = start / 43
            is_drop = 0
            
            if track_name in drops_info:
                drop_start, drop_end = drops_info[track_name]
                if drop_start <= current_sec <= drop_end:
                    is_drop = 1
                
            row = [mean_rms, std_rms, mean_cent, std_cent, mean_zcr]
            data_set_x.append(row)
            data_set_y.append(is_drop)
                
    return data_set_x, data_set_y

