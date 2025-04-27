from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import librosa
import matplotlib.pyplot as plt

def create_pdf(name, score, spoof_result, datetime_now, wav_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "Nutqni aniqlash hisobot")

    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, height - 3 * cm, f"Ismi: {name}")
    c.drawString(2 * cm, height - 4 * cm, f"O‘xshashlik: {round(score * 100, 1)}%")
    c.drawString(2 * cm, height - 5 * cm, f"Ovoz haqiqiyligi: {spoof_result}")
    c.drawString(2 * cm, height - 6 * cm, f"Sana va vaqt: {datetime_now}")

    # Grafik chizish
    y, sr = librosa.load(wav_path, sr=16000)
    plt.figure(figsize=(6, 2))
    plt.plot(y)
    plt.title("Ovoz signali grafigi")
    plt.xlabel("Namuna raqami")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    graph_path = "static/signal.png"
    plt.savefig(graph_path)
    plt.close()

    c.drawImage(graph_path, 2 * cm, height - 15 * cm, width=14*cm, height=7*cm)

    c.showPage()
    c.save()
