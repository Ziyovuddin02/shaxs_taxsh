from flask import Flask, request, render_template, jsonify, send_file
import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64
from spoof_detector import is_spoof
from generate_pdf import create_pdf
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
KNOWN_FOLDER = "known_voices"
PDF_FOLDER = "pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

encoder = VoiceEncoder()
known_speakers = {}

# Oldindan tanilgan shaxslar yuklanadi
def load_known_speakers():
    for fname in os.listdir(KNOWN_FOLDER):
        if fname.endswith((".wav", ".mp3", ".mp4")):
            path = Path(os.path.join(KNOWN_FOLDER, fname))
            wav_path = convert_to_wav(str(path))
            wav = preprocess_wav(Path(wav_path))
            embedding = encoder.embed_utterance(wav)
            name = os.path.splitext(fname)[0]
            known_speakers[name] = embedding

# Har qanday audio/video faylni .wav formatga o'tkazish
def convert_to_wav(file_path):
    wav_path = file_path
    if file_path.endswith((".mp3", ".mp4")):
        sound = AudioSegment.from_file(file_path)
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
    return wav_path

# Spektrogram yaratish
def generate_spectrogram(wav_path):
    y, sr = librosa.load(wav_path)
    plt.figure(figsize=(6, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    try:
        file = request.files['audio']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        wav_path = convert_to_wav(file_path)

        # WAVni yuklab tekshiramiz
        y, sr = librosa.load(wav_path, sr=16000)

        # 1. Ovoz uzunligi tekshirish (kamida 1 sekund bo'lishi kerak)
        if len(y) < sr:
            return jsonify({"error": "Ovoz juda qisqa. Kamida 1 soniyalik ovoz yuboring."})

        # 2. Ovoz energiyasini tekshirish (0.001 dan katta bo‘lishi kerak)
        energy = np.sum(y ** 2) / len(y)
        if energy < 0.001:
            return jsonify({"error": "Ovoz kuchi juda past. Iltimos, aniq gapiring."})

        # 3. Agar hammasi OK bo'lsa - Embedding olamiz
        wav = preprocess_wav(Path(wav_path))
        test_embedding = encoder.embed_utterance(wav)

        best_match = None
        best_score = -1
        for name, emb in known_speakers.items():
            score = np.dot(test_embedding, emb) / (np.linalg.norm(test_embedding) * np.linalg.norm(emb))
            if score > best_score:
                best_score = score
                best_match = name

        # 4. Spoof (deepfake) tekshiruv
        spoof = is_spoof(wav_path)

        # 5. Spektrogram va PDF
        spectrogram_img = generate_spectrogram(wav_path)
        now = datetime.now().strftime("%d-%B-%Y, %H:%M")
        pdf_path = os.path.join(PDF_FOLDER, f"{Path(file.filename).stem}.pdf")
        create_pdf(best_match, best_score, spoof, now, wav_path, pdf_path)

        return jsonify({
            "name": best_match,
            "score": round(float(best_score) * 100, 1),
            "spoof": spoof,
            "spectrogram": spectrogram_img,
            "pdf_link": f"/download/{Path(pdf_path).name}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/download/<filename>')
def download_pdf(filename):
    return send_file(os.path.join(PDF_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    load_known_speakers()
    app.run(debug=True)
