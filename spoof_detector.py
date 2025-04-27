import librosa
import numpy as np

def is_spoof(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    energy = np.sum(y ** 2) / len(y)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Oddiy heuristika: past energiya + yuqori ZCR + baland spectral centroid = Deepfake ehtimoli
    if energy < 0.0003 and zero_crossing_rate > 0.1 and spectral_centroid > 3000:
        return "❌ Klonlangan (Deepfake ehtimoli bor)"
    else:
        return "✅ Haqiqiy ovoz"
