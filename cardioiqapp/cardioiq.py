import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import scipy.io.wavfile as wav
from fpdf import FPDF
from datetime import datetime
from ecg_model import Net
from risk_score import calculate_cardiovascular_risk
from torchcam.methods import GradCAM
import tempfile
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


def preprocess_wav(file):
    fs_target = 360  
    sr, signal = wav.read(file)
    signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = signal[:,0]
    if sr != fs_target:
        signal = np.interp(np.linspace(0, len(signal)-1, fs_target*2), np.arange(len(signal)), signal)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    if len(signal) < 720:
        signal = np.pad(signal, (0, 720 - len(signal)))
    else:
        signal = signal[:720]
    return signal[np.newaxis, :]  # (1,720)

def predict_ecg(model, signal, threshold=0.5):
    ecg_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,1,720)
    ecg_tensor.requires_grad = True

    cam_extractor = GradCAM(model, target_layer=model.encoder[-1].conv2)
    output = model(ecg_tensor)
    prob = torch.sigmoid(output).item()
    pred = int(prob > threshold)

    cam_list = cam_extractor(class_idx=0, scores=output)
    cam = cam_list[0].detach().cpu().numpy()
    cam = np.interp(cam, (cam.min(), cam.max() + 1e-6), (0, 1)).squeeze()

    ecg = ecg_tensor.squeeze().detach().cpu().numpy()
    cam_upsampled = np.interp(np.arange(len(ecg)), np.linspace(0, len(ecg)-1, num=len(cam)), cam)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(ecg, color='blue')
    ax.fill_between(np.arange(len(ecg)), 0, ecg, where=(cam_upsampled > 0.3), color='red', alpha=0.4)
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    title = "ECG: Abnormal" if pred else "ECG: Normal"
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    return pred, prob, fig

def evaluate_messages(hr, bp_sys, bp_dia, spo2, smoking_freq, alcohol_per_week, sleep_hours, stress_level, pred):
    messages = []

    if hr > 100:
        messages.append(("High heart rate (tachycardia)", "red"))
    elif hr < 60:
        messages.append(("Low heart rate (bradycardia)", "orange"))
    else:
        messages.append(("Heart rate normal", "green"))

    if bp_sys >= 140 or bp_dia >= 90:
        messages.append(("Hypertension detected", "red"))
    elif bp_sys < 90 or bp_dia < 60:
        messages.append(("Low blood pressure", "orange"))
    else:
        messages.append(("Blood pressure normal", "green"))

    if spo2 < 94:
        messages.append(("Low oxygen saturation", "red"))
    else:
        messages.append(("SpO₂ normal", "green"))

    if smoking_freq in ["Weekly", "Daily"]:
        messages.append(("Smoking habit detected", "red"))
    if alcohol_per_week > 7:
        messages.append(("High alcohol intake", "orange"))
    if sleep_hours < 6:
        messages.append(("Insufficient sleep", "orange"))
    if stress_level > 7:
        messages.append(("High stress level", "orange"))

    if pred:
        messages.append(("ECG abnormal", "red"))
    else:
        messages.append(("ECG normal", "green"))

    return messages
def _clean_pdf(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    repl = {
        "–": "-", "—": "-", "−": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "°": " deg", "µ": "u", "•": "-",
        "×": "x", "…": "...", "™": "(TM)", "©": "(c)", "®": "(R)",

        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
   
    return text.encode("latin-1", "replace").decode("latin-1")

def _pdf_write_multiline(pdf, text: str):
    for line in _clean_pdf(text).split("\n"):
        pdf.multi_cell(0, 8, line)

def save_pdf(report_text, fig, ecg_label, messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    _pdf_write_multiline(pdf, report_text)

    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=14)
    pdf.multi_cell(0, 8, _clean_pdf(f"ECG Result: {ecg_label}"))

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, _clean_pdf("Messages:"), ln=True)
    for msg, color in messages:
        if color == "red":
            pdf.set_text_color(255, 0, 0)
        elif color == "orange":
            pdf.set_text_color(255, 165, 0)
        else:
            pdf.set_text_color(0, 128, 0)
        pdf.multi_cell(0, 8, _clean_pdf(f"- {msg}"))
    pdf.set_text_color(0, 0, 0)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        fig.savefig(tmp_img.name, bbox_inches="tight")
        tmp_img_path = tmp_img.name
    pdf.image(tmp_img_path, x=15, w=180)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf_path = tmp_pdf.name

    with open(tmp_pdf_path, "rb") as f:
        pdf_bytes = io.BytesIO(f.read())

    os.remove(tmp_img_path)
    os.remove(tmp_pdf_path)

    pdf_bytes.seek(0)
    return pdf_bytes


st.title("CardioIQ")
st.header("Patient Information")

if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 120, 25)
    if st.button("Start"):
        if name.strip():
            st.session_state.start = True
            st.session_state.name = name
            st.session_state.age = age
        else:
            st.warning("Enter a name to start")
    st.stop()

st.header("Step 1: Enter Vitals & Lifestyle")
col1, col2 = st.columns(2)

with col1:
    hr = st.number_input("Heart Rate (bpm)", 30, 200, 78)
    bp_sys = st.number_input("Systolic BP (mmHg)", 80, 200, 132)
    bp_dia = st.number_input("Diastolic BP (mmHg)", 50, 130, 90)
    spo2 = st.number_input("SpO2 (%)", 80, 100, 99)
    weight_kg = st.number_input("Weight (kg)", 30, 200, 70)
    height_cm = st.number_input("Height (cm)", 120, 220, 175)

with col2:
    smoking_freq = st.selectbox("Smoking Frequency", ["Never", "Occasional", "Weekly", "Daily"], index=0)
    alcohol_per_week = st.slider("Alcohol (drinks/week)", 0, 14, 2)
    diet_score = st.slider("Diet Score (0=worst,10=best)", 0, 10, 5)
    sleep_hours = st.slider("Average Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
    stress_level = st.slider("Stress Level (0-10)",0,10,3)
    family_history = st.checkbox("Family history of heart disease?")

st.header("Step 2: Upload ECG (.wav)")
ecg_file = st.file_uploader("Upload a 2-second ECG .wav file", type=["wav"])

if st.button("Run Diagnostic") and ecg_file is not None:
    ecg_signal = preprocess_wav(ecg_file)
    pred, prob, fig = predict_ecg(model, ecg_signal)
    ecg_label = "Abnormal" if pred else "Normal"

    sex_val = 0
    risk_score, risk_category = calculate_cardiovascular_risk(
        hr, bp_sys, bp_dia, spo2,
        ["Never", "Occasional", "Weekly", "Daily"].index(smoking_freq),
        alcohol_per_week, 10-diet_score, sleep_hours, stress_level,
        st.session_state.age, sex_val, family_history, pred,
        weight_kg, height_cm
    )

    messages = evaluate_messages(hr, bp_sys, bp_dia, spo2, smoking_freq, alcohol_per_week, sleep_hours, stress_level, pred)

    st.header("Results")
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Cardiovascular Risk", f"{risk_score:.1f}%")
        st.write(f"Risk Category: **{risk_category}**")
        for msg, color in messages:
            st.markdown(f"<span style='color:{color}'>{msg}</span>", unsafe_allow_html=True)
    with col2:
        st.pyplot(fig)

    report_text = f"""
Patient: {st.session_state.name}
Age: {st.session_state.age}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- Vitals & Lifestyle ---
HR: {hr} bpm | BP: {bp_sys}/{bp_dia} mmHg | SpO2: {spo2}%
Weight: {weight_kg} kg | Height: {height_cm} cm
Smoking: {smoking_freq} | Alcohol/week: {alcohol_per_week}
Diet Score: {diet_score} | Sleep: {sleep_hours} hrs | Stress: {stress_level}/10
Family history: {"Yes" if family_history else "No"}

--- Cardiovascular Risk ---
Score: {risk_score:.1f}% | Category: {risk_category}
"""

    pdf_buffer = save_pdf(report_text, fig, ecg_label, messages)
    st.download_button("Download Report (PDF)", data=pdf_buffer, file_name="cardio_report.pdf", mime="application/pdf")
else:
    if ecg_file is None:
        st.info("Upload an ECG .wav file to run diagnostic")
