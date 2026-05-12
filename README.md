# 🎵 Drop Detector — AI-Powered Music Drop Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-orange?style=for-the-badge)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Analysis-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**Automatic detection of epic moments ("drops") in music tracks using Machine Learning and audio feature extraction.**

</div>

---

# 📌 Overview

**Drop Detector** is an AI-powered audio analysis tool that automatically detects the climax or **drop** in music tracks.

The system uses:

- 🎧 Acoustic feature extraction
- 🧠 Machine Learning (SVM Classifier)
- 📊 Signal processing techniques

to identify high-energy moments commonly found in:

- Phonk
- EDM
- Dubstep
- Trap
- Bass Music

---

### System Requirements
For audio processing to work, you must have **FFmpeg** installed on your system.
* **Windows:** `winget install ffmpeg` (or download from official site)
* **macOS:** `brew install ffmpeg`
* **Linux:** `sudo apt install ffmpeg`

---

# ✨ Features

- ✅ Automatic drop detection
- ✅ Audio waveform analysis
- ✅ ML-based prediction scoring
- ✅ Confidence estimation
- ✅ 2-second sliding window analysis
- ✅ Optimized dependency management via `uv`
- ✅ WAV lossless audio support

---

# 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12+ | Core language |
| librosa | Audio analysis & DSP |
| pandas | Data processing |
| scikit-learn | SVM classifier |
| yt-dlp | Audio downloading |
| ffmpeg | Audio conversion/backend |
| uv | Dependency management |

---

# 📦 Installation

## 1️⃣ Install uv

Visit:

👉 https://github.com/astral-sh/uv

---

## 2️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/drop-detector.git
cd drop-detector
```

---

## 3️⃣ Install Dependencies

```bash
uv sync
```

This command will:

- Create a `.venv`
- Install all dependencies
- Resolve versions using `uv.lock`

---

# 🎧 Prepare Audio Files

Place your `.wav` tracks inside:

```plaintext
data/test/
```

> ⚠️ Recommended format: `.wav`
>
> Lossless audio significantly improves detection accuracy.

---

# 🚀 Run the Detector

Execute:

```bash
python scripts/main.py
```

Example output:

```plaintext
[TRACK] NIGHT DRIVE.wav
Best Drop: 01:12
Confidence: 96.4%
```

---

# 📂 Project Structure

```plaintext
.
├── bin/                  # System binaries (ffmpeg)
├── data/
│   ├── raw/              # Training dataset
│   └── test/             # Tracks for detection
├── models/               # Trained ML models
│   ├── drop_model.pkl
│   └── scaler.pkl
├── scripts/
│   ├── main.py
│   └── train_detector.py
├── src/                  # Core source code
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 🧠 How It Works

The detector splits audio into **2-second windows** and extracts acoustic features from each segment.

## Extracted Features

| Feature | Description |
|---|---|
| Mean RMS | Overall loudness |
| Std RMS | Volume dynamics |
| Spectral Centroid | Brightness of sound |
| Std Spectral Centroid | Frequency movement |
| Zero Crossing Rate | Percussion/noise intensity |
| Spectral Rolloff | High-frequency energy |

---

# 🤖 Machine Learning Model

The project uses:

- **Support Vector Machine (SVM)**
- **RBF Kernel**
- **StandardScaler normalization**

The classifier learns to separate:

✅ Normal track segments  
from  
🔥 High-energy drop moments

---

# 📈 Detection Pipeline

```plaintext
Audio Track
    ↓
Window Slicing (2 sec)
    ↓
Feature Extraction
    ↓
Feature Normalization
    ↓
SVM Prediction
    ↓
Drop Confidence Score
```

---

# 🔥 Supported Genres

- Phonk
- EDM
- Dubstep
- Trap
- Hardwave
- Bass House
- Experimental Bass

---

# 📋 Requirements

- Python `>=3.12`
- ffmpeg installed and available in PATH

---

# 🧪 Training Your Own Model

To retrain the detector:

```bash
python scripts/train_detector.py
```

Make sure your labeled dataset is stored in:

```plaintext
data/raw/
```

---

# 📄 License

MIT License

---

</div>
