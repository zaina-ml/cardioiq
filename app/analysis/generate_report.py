import io
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

from config import PDF_CONFIG
from analysis.analysis import generate_findings


def generate_profile_pdf(profile, results_sorted):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=PDF_CONFIG["MARGIN"])

    pdf.add_page()
    pdf.set_font(PDF_CONFIG["FONT"], "B", 16)
    pdf.cell(0, 10, f"CardioIQ Profile Report: {profile['name']}", ln=True, align="C")

    pdf.set_font(PDF_CONFIG["FONT"], "", 12)
    bmi = round(profile['weight_kg'] / ((profile['height_cm'] / 100) ** 2), 1)
    pdf.ln(5)
    pdf.cell(
        0, 8,
        f"Sex: {profile['sex']} | Age: {profile['age']} | BMI: {bmi}",
        ln=True,
        align="C"
    )

    ecg_probs = [r["ecg_abnormality_prob"] for r in results_sorted]
    risk_scores = [r["cardio_risk_score"] for r in results_sorted]

    avg_ecg = np.mean(ecg_probs)
    avg_risk = np.mean(risk_scores)

    pdf.ln(10)
    pdf.set_font(PDF_CONFIG["FONT"], "B", 14)
    pdf.cell(0, 10, "Summary Metrics", ln=True)

    pdf.set_font(PDF_CONFIG["FONT"], "", 12)
    pdf.cell(0, 8, f"Average ECG Abnormality Probability: {avg_ecg:.2f}", ln=True)
    pdf.cell(0, 8, f"Average Cardiovascular Risk: {avg_risk:.2%}", ln=True)

    timestamps = [pd.to_datetime(r["timestamp"]) for r in results_sorted]

    def add_plot(fig):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.write_image(tmp.name)
            pdf.add_page()
            pdf.image(tmp.name, x=15, w=PDF_CONFIG["IMG_WIDTH"])
        os.unlink(tmp.name)

    fig_risk = go.Figure(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode="lines+markers"
    ))
    fig_risk.update_layout(title="Risk Score Over Time", yaxis=dict(range=[0, 1]))
    add_plot(fig_risk)

    fig_ecg = go.Figure(go.Scatter(
        x=timestamps,
        y=ecg_probs,
        mode="lines+markers"
    ))
    fig_ecg.update_layout(title="ECG Abnormality Probability Over Time", yaxis=dict(range=[0, 1]))
    add_plot(fig_ecg)

    for i, r in enumerate(results_sorted, 1):
        pdf.add_page()
        pdf.set_font(PDF_CONFIG["FONT"], "B", 14)
        pdf.cell(0, 10, f"Result {i} | {r['timestamp'][:19]}", ln=True)

        pdf.set_font(PDF_CONFIG["FONT"], "", 12)
        pdf.cell(0, 8, f"ECG Prediction: {r['ecg_prediction']} | Probability: {r['ecg_abnormality_prob']:.2f}", ln=True)
        pdf.cell(0, 8, f"Cardiovascular Risk: {r['cardio_risk_score']:.2%}", ln=True)

        pdf.ln(5)
        pdf.set_font(PDF_CONFIG["FONT"], "B", 12)
        pdf.cell(0, 8, "Findings:", ln=True)

        findings = generate_findings(r, avg_risk)

        pdf.set_font(PDF_CONFIG["FONT"], "", 12)
        for text, severity in findings:
            if severity == 3:
                pdf.set_text_color(255, 0, 0)
            elif severity == 2:
                pdf.set_text_color(255, 165, 0)
            else:
                pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 6, f"- {text}")

        pdf.set_text_color(0, 0, 0)

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return io.BytesIO(pdf_bytes)
