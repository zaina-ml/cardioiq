import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import io
import zipfile
from fpdf import FPDF
import time
import os
import uuid
from datetime import datetime
import tempfile
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

from ecgnet import ECGNet
from cardiorisknet import CardioRiskNet
from ecg_wgan import ECGGenerator
from gradcam import GradCAM, plot_ecg_cam
from generate_report import generate_profile_pdf
from analysis import generate_findings
from config import THRESHOLDS, MODEL_THRESHOLDS

if "active_profile" not in st.session_state:
    st.session_state.active_profile = None

firebase_config = dict(st.secrets["firebase"])
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

def init_firebase_admin():
    if not firebase_admin._apps:
        cred = credentials.Certificate(".streamlit/service_account.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase_admin()

st.set_page_config(page_title="CardioIQ", layout="wide", initial_sidebar_state="expanded")
st.title("CardioIQ")

ECG_MODEL_PATH = "models/ecgnet_model.pt"
CARDIORISK_MODEL_PATH = "models/cardiorisknet_model.pt"
ECG_WGAN_PATH = "models/ecg_wgan_generator.pt"


@st.cache_resource
def load_ecg_model():
    model = ECGNet()
    state = torch.load(ECG_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    target_layer = model.r3.bn2
    gradcam = GradCAM(model, target_layer)
    return model, gradcam

@st.cache_resource
def load_risk_model(model_path=CARDIORISK_MODEL_PATH):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = CardioRiskNet(input_size=11)
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_ecg_wgan_model(model_path=ECG_WGAN_PATH):
    state_dict = torch.load(model_path, map_location="cpu")
    model = ECGGenerator()
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model, gradcam = load_ecg_model()
    cardiorisk_model = load_risk_model()
    ecg_wgan_model = load_ecg_wgan_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

def login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state['user'] = {"localId": user.get("localId"), "email": user.get("email")}
        return True, None
    except Exception as e:
        return False, str(e)

def signup(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state['user'] = {"localId": user.get("localId"), "email": user.get("email")}
        return True, None
    except Exception as e:
        return False, str(e)

def logout():
    st.session_state.pop('user', None)

def save_result(user_uid, profile, filename, ecg_prob, risk_score, features):
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "profile_id": profile["id"],
        "profile_name": profile["name"],

        "filename": filename,
        "ecg_abnormality_prob": float(ecg_prob),
        "ecg_prediction": "Abnormal" if ecg_prob >= MODEL_THRESHOLDS["THRESHOLD"] else "Normal",
        "cardio_risk_score": float(risk_score),
        "features": features
    }

    db.collection("users") \
      .document(user_uid) \
      .collection("profiles") \
      .document(profile["id"]) \
      .collection("results") \
      .add(data)

def profile_name_exists(profiles, name, exclude_id=None):
    name = name.strip().lower()
    for p in profiles:
        if exclude_id and p["id"] == exclude_id:
            continue
        if p["name"].strip().lower() == name:
            return True
    return False

def switch_profile():
    selected_label = st.session_state.active_profile_select
    st.session_state.active_profile = profile_map[selected_label]
    for k in [
        "ecg", "ecg_tensor", "ecg_prob", "ecg_cam", "risk_score",
        "uploaded_filename", "source_label", "ecg_label", "fake_ecg"
    ]:
        st.session_state.pop(k, None)
    st.toast(f"Switched to profile: {selected_label}")

if "user" not in st.session_state:
    auth_mode = st.sidebar.radio(
        "Select Option", ["Login", "Sign Up"], key="auth_mode_radio"
    )
    email = st.sidebar.text_input("Email", key="auth_email")
    password = st.sidebar.text_input("Password", type="password", key="auth_password")

    if auth_mode == "Login" and st.sidebar.button("Login", key="login_btn"):
        success, msg = login(email, password)
        if success:
            st.success("Logged in!")
            st.rerun()
        else:
            st.error(f"Login Failed: {msg}")

    elif auth_mode == "Sign Up" and st.sidebar.button("Sign Up", key="signup_btn"):
        success, msg = signup(email, password)
        if success:
            st.success("Account created and logged in!")
            st.rerun()
        else:
            st.error(f"Signup Failed: {msg}")

else:
    st.sidebar.success(f"Logged in as {st.session_state['user']['email']}")
    user_uid = st.session_state["user"]["localId"]

    profiles_ref = db.collection("users").document(user_uid).collection("profiles")
    profiles = [p.to_dict() | {"id": p.id} for p in profiles_ref.stream()]

    if profiles and "active_profile" not in st.session_state:
        st.session_state.active_profile = profiles[0]
    elif not profiles:
        st.session_state.pop("active_profile", None)

    if profiles:
        profile_map = {p['name']: p for p in profiles}
        profile_names = list(profile_map.keys())

        if st.session_state.get("profile_adj"):
            st.session_state.profile_adj = False
            st.session_state.active_profile_select = st.session_state.active_profile["name"]


        st.sidebar.selectbox(
            "Active profile",
            profile_names,
            key="active_profile_select",
            on_change=switch_profile
        )

        if "active_profile" not in st.session_state:
            st.session_state.active_profile = profile_map[profile_names[0]]

    else:
        st.sidebar.info("No profiles yet")

    if st.session_state.get("active_profile"):
        profile = st.session_state.active_profile
        pid = profile["id"]

        with st.sidebar.expander("Edit active profile"):
            new_name = st.text_input("Name", profile["name"], key=f"edit_name_{pid}")
            new_age = st.number_input("Age", 0, 120, profile["age"], key=f"edit_age_{pid}")
            new_sex = st.selectbox(
                "Sex", ["Male", "Female"],
                index=0 if profile["sex"] == "Male" else 1,
                key=f"edit_sex_{pid}"
            )
            new_height = st.number_input("Height (cm)", 100, 210, profile["height_cm"], key=f"edit_height_{pid}")
            new_weight = st.number_input("Weight (kg)", 10.0, 200.0, profile["weight_kg"], key=f"edit_weight_{pid}")

            if st.button("Save changes", key=f"save_profile_{pid}"):
                if not new_name.strip():
                    st.error("Name cannot be empty.")
                elif profile_name_exists(profiles, new_name, exclude_id=pid):
                    st.error("Another profile already has this name.")
                else:
                    profiles_ref.document(pid).update({
                        "name": new_name.strip(),
                        "age": new_age,
                        "sex": new_sex,
                        "height_cm": new_height,
                        "weight_kg": new_weight,
                        "updated_at": datetime.utcnow().isoformat()
                    })

                    st.session_state.active_profile.update({
                        "name": new_name.strip(),
                        "age": new_age,
                        "sex": new_sex,
                        "height_cm": new_height,
                        "weight_kg": new_weight
                    })

                    st.session_state.profile_adj = True

                    st.toast("Profile updated.")
                    st.rerun()

    with st.sidebar.expander("Add new profile"):
        name = st.text_input("Name", key="create_name")
        age = st.number_input("Age", 0, 120, 40, key="create_age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="create_sex")
        height = st.number_input("Height (cm)", 100, 210, 170, key="create_height")
        weight = st.number_input("Weight (kg)", 10.0, 200.0, 70.0, key="create_weight")

        if st.button("Create profile", key="create_profile_btn"):
            if not name.strip():
                st.error("Name is required.")
            elif profile_name_exists(profiles, name):
                st.error("A profile with this name already exists.")
            else:
                _, doc_ref = profiles_ref.add({
                    "name": name.strip(),
                    "age": age,
                    "sex": sex,
                    "height_cm": height,
                    "weight_kg": weight,
                    "created_at": datetime.utcnow().isoformat()
                })

                st.session_state.active_profile = {
                    "id": doc_ref.id,
                    "name": name.strip(),
                    "age": age,
                    "sex": sex,
                    "height_cm": height,
                    "weight_kg": weight
                }

                st.session_state.profile_adj = True

                st.toast("Profile created.")
                st.rerun()

        if st.session_state.get("active_profile"):
            with st.sidebar.expander("Delete active profile"):
                confirm = st.checkbox(
                    f"I understand deleting '{profile['name']}' cannot be undone.",
                    key=f"delete_confirm_{pid}"
                )

                if st.button("Delete profile", disabled=not confirm, key=f"delete_profile_{pid}"):
                    results_ref = profiles_ref.document(pid).collection("results")
                    for doc in results_ref.stream():
                        doc.reference.delete()

                    profiles_ref.document(pid).delete()

                    for k in [
                        "ecg", "ecg_tensor", "ecg_prob", "ecg_cam", "risk_score",
                        "uploaded_filename", "source_label", "ecg_label", "fake_ecg"
                    ]:
                        st.session_state.pop(k, None)

                    remaining = [p for p in profiles if p["id"] != pid]
                    if remaining:
                        st.session_state.active_profile = remaining[0]
                        st.toast(f"Switched to profile: {remaining[0]['name']}")
                    else:
                        st.session_state.pop("active_profile", None)
                        st.sidebar.info("No profiles remaining")

                    st.rerun()
                    
    if st.sidebar.button("Logout", key="logout_btn"):
        logout()
        st.rerun()


tabs = st.tabs(["Predict", "Profile", "About"])

with tabs[2]:
    st.header("About CardioIQ")

    st.markdown("""
    **CardioIQ** is an experimental, AI-driven cardiovascular analysis platform designed to 
    explore how deep learning models can integrate physiological signals with lifestyle and 
    clinical risk factors.

    The system combines automated ECG interpretation with a multi-factor cardiovascular risk 
    model, providing transparent model outputs and longitudinal trend tracking at the 
    individual profile level.
    """)

    st.markdown("### Core Capabilities")

    st.markdown("""
    • **ECGNet**: Deep residual SE convolutional network trained on the MIT-BIH Arrhythmia Database for 
      ECG abnormality detection \n
    • **Explainability**: Grad-CAM–based visualization highlighting signal regions that 
      influence model predictions\n
    • **Synthetic Signal Generation**: wGAN for experimental ECG simulation\n
    • **Risk Modeling**: Multi-input MLP integrating ECG output with demographics, lifestyle, 
      and blood pressure metrics\n
    • **Longitudinal Tracking**: Per-profile result history with baseline comparison and 
      trend analysis\n  
    • **Secure Architecture**: Firebase-based authentication and per-user data isolation\n
    """)

st.divider()
st.caption("Zain Aboobacker · 2025")
st.caption("This tool should not be used as a substitute for professional medical advice.")

with tabs[0]:
    if 'user' not in st.session_state:
        st.info("Please login or sign up to use the app.")
    else:
        profile = st.session_state.get("active_profile")

        if not profile:
            st.warning("Create or select a profile to continue.")
            st.stop()

        st.caption(
            f"Active profile: {profile['name']} "
            f"({profile['sex']}, {profile['age']}y)"
        )

        col_left, col_right = st.columns(2)
        window_size = 720

        with col_left:
            st.write("### Import ECG CSV")
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                if ('uploaded_filename' not in st.session_state or
                        st.session_state['uploaded_filename'] != uploaded_file.name):
                    ecg_data = np.loadtxt(uploaded_file, delimiter=",")
                    ecg_data = (ecg_data - ecg_data.mean()) / (ecg_data.std() + 1e-8)
                    ecg_data = np.pad(ecg_data, (0, max(0, window_size - len(ecg_data))))[:window_size]
                    st.session_state['ecg'] = ecg_data
                    st.session_state['uploaded_filename'] = uploaded_file.name
                    st.session_state['ecg_tensor'] = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    st.session_state['ecg_prob'] = None
                    st.session_state['ecg_cam'] = None
                    st.session_state.pop('risk_score', None)
                    st.session_state.pop('fake_ecg', None)
                    st.session_state.source_label = "CSV"
            else:
                if st.session_state.get('source_label') == "CSV":
                    st.session_state.pop('ecg_tensor', None)
                    st.session_state.pop('ecg_prob', None)
                    st.session_state.pop('ecg_cam', None)
                    st.session_state.pop('risk_score', None)
                    st.session_state.pop('uploaded_filename', None)
                    st.session_state.pop('source_label', None)

        with col_right:
            st.write("### Simulated ECG Device")
            st.markdown("<small>Generate a synthetic ECG signal using a WGAN model (Experimental)</small>", unsafe_allow_html=True)

            disable_generate = st.session_state.get('source_label') == "CSV"
            generate = st.button("Record Simulated ECG", key="simulate_ecg", use_container_width=True, disabled=disable_generate)
            if disable_generate:
                st.caption("Cannot record simulated ECG because a CSV is uploaded.")
            if generate:
                with torch.inference_mode():
                    ecg_wgan_model.eval()
                    z = torch.randn(1, 100)
                    ecg_fake = ecg_wgan_model(z).detach().cpu().numpy().squeeze()

                st.session_state['ecg'] = ecg_fake
                st.session_state['ecg_tensor'] = torch.tensor(ecg_fake, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                st.session_state['ecg_prob'] = None
                st.session_state['ecg_cam'] = None
                st.session_state.pop('risk_score', None)
                st.session_state.source_label = "Simulated ECG"

        if st.session_state.get('ecg_tensor') is not None:
            if generate:
                warning = st.empty()
                progress = st.empty()

                with st.spinner("Connecting to ECG device..."):
                    time.sleep(2.0)

                warning.warning("Recording ECG...")
                progress = progress.progress(0)

                duration_sec = 2.0
                steps = 20

                for i in range(steps):
                    percent = int((i + 1) / steps * 100)
                    progress.progress(percent)
                    time.sleep(duration_sec / steps)

                warning.empty()
                progress.empty()

                with st.spinner("Processing ECG…"):
                    time.sleep(1.0)

                st.toast("ECG captured successfully")

                fig = go.Figure(go.Scatter(y=st.session_state['ecg'], mode="lines", line=dict(color="red")))
                fig.update_layout(
                    title=f"ECG Preview ({st.session_state.get('source_label', '')})",
                    xaxis_title="Samples",
                    yaxis_title="Amplitude",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure(go.Scatter(y=st.session_state['ecg'], mode="lines", line=dict(color="red")))
                fig.update_layout(
                    title=f"ECG Preview ({st.session_state.get('source_label', '')})",
                    xaxis_title="Samples",
                    yaxis_title="Amplitude",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("ECGNet Prediction")

        analyze_disabled = st.session_state.get('ecg_tensor') is None
        if st.button("Analyze ECG", disabled=analyze_disabled):
            with st.spinner("Analyzing ECG..."):
                ecg_tensor = st.session_state['ecg_tensor']
                with torch.inference_mode():
                    st.session_state['ecg_prob'] = float(torch.sigmoid(model(ecg_tensor)).item())
                cam, _ = gradcam(ecg_tensor)
                st.session_state['ecg_cam'] = cam.squeeze()
                st.session_state['ecg_label'] = "Abnormal" if st.session_state['ecg_prob'] >= MODEL_THRESHOLDS["THRESHOLD"] else "Normal"

        if st.session_state.get('ecg_prob') is not None:
    
            st.markdown(
                f"<p class='{ 'big-font' if st.session_state.ecg_label=='Abnormal' else 'normal-font' }'>Prediction: {st.session_state.ecg_label}</p>",
                unsafe_allow_html=True
            )
            st.metric(label="Abnormal Probability", value=f"{st.session_state.ecg_prob:.2f}")

            fig_cam = plot_ecg_cam(st.session_state['ecg_tensor'], st.session_state['ecg_cam'], st.session_state['ecg_label'])
            st.plotly_chart(fig_cam, use_container_width=True)
        
        st.divider()

        st.subheader("Cardiovascular Risk Prediction")
        st.markdown("*(Only computed after ECG analysis)*")

        age = profile["age"]
        sex = profile["sex"]

        height_cm = profile["height_cm"]
        weight_kg = profile["weight_kg"]

        exercise = st.slider("Exercise Level (0=Sedentary, 1=Active)", 0.0, 1.0, 0.5, help="How active the person is on average. 0 = sedentary, 1 = fully active")
        diet = st.slider("Diet Quality (0=Poor, 1=Ideal)", 0.0, 1.0, 0.5, help="Overall diet quality. 0 = poor, 1 = ideal")
        sleep = st.slider("Sleep Quality (0=Poor, 1=Ideal)", 0.0, 1.0, 0.5, help="Quality of sleep. 0 = poor, 1 = ideal")

        smoking_per_week = st.slider("Cigarettes per Week", 0, 140, 0, step=1, help="Number of cigarettes smoked per week")
        alcohol_per_week = st.slider("Alcoholic Drinks per Week", 0, 30, 0, step=1, help="Number of alcoholic drinks per week")

        sys_bp = st.number_input("Systolic BP (mmHg)", 50, 220, 120, help="Systolic blood pressure in mmHg")
        dia_bp = st.number_input("Diastolic BP (mmHg)", 50, 200, 80, help="Diastolic blood pressure in mmHg")

        disable_risk = st.session_state.get('ecg_prob') is None
        if st.button("Compute Cardiovascular Risk", disabled=disable_risk):
            with st.spinner("Computing cardiovascular risk..."):
                ecg_prob = st.session_state['ecg_prob']

                age_norm = (age - 20) / 60
                height_m = height_cm / 100.0
                bmi = weight_kg / (height_m ** 2)

                bmi_norm = (bmi - 18.5) / 16.5
                bmi_norm = max(0.0, min(bmi_norm, 1.0))
                sys_bp_norm = (sys_bp - 90) / 90
                dia_bp_norm = (dia_bp - 60) / 60
                smoking_norm = smoking_per_week / 140
                alcohol_norm = alcohol_per_week / 30
                sex_val = 1 if sex == "Female" else 0
                
                risk_features = torch.tensor([
                    ecg_prob,
                    exercise,
                    diet,
                    sleep,
                    smoking_norm,
                    alcohol_norm,
                    age_norm,
                    sex_val,
                    bmi_norm,
                    sys_bp_norm,
                    dia_bp_norm
                ], dtype=torch.float32).unsqueeze(0)

                with torch.inference_mode():
                    risk_score = float(cardiorisk_model(risk_features).item())
                    st.session_state['risk_score'] = risk_score

            st.metric("Predicted Cardiovascular Risk", f"{risk_score:.2%}")
            st.toast("Check the profile tab to learn more about your result.")

            user_uid = st.session_state['user']['localId']
            user_email = st.session_state['user']['email']
            filename = (
                st.session_state.get('uploaded_filename')
                if st.session_state['source_label'] == "CSV"
                else "Simulated_ECG"
            )

            features = {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "exercise": exercise,
                "diet": diet,
                "sleep": sleep,
                "smoking_per_week": smoking_per_week,
                "alcohol_per_week": alcohol_per_week,
                "sys_bp": sys_bp,
                "dia_bp": dia_bp
            }

            save_result(user_uid, profile, filename, ecg_prob, risk_score, features)


with tabs[1]:
    if 'user' not in st.session_state:
        st.info("Please login to view your profile.")
        st.stop()

    profile = st.session_state.active_profile

    if not profile:
        st.warning("Select a profile to view history.")
        st.stop()

    st.subheader(f"Profile Overview: {profile['name']}")

    user_uid = st.session_state['user']['localId']
    results_ref = (
        db.collection("users")
        .document(user_uid)
        .collection("profiles")
        .document(profile["id"])
        .collection("results")
    )

    results = [r.to_dict() | {"id": r.id} for r in results_ref.stream()]

    if not results:
        st.info("No past results yet. Run an analysis in the Predict tab.")
        st.stop()

    ecg_probs = [r["ecg_abnormality_prob"] for r in results]
    risk_scores = [r["cardio_risk_score"] for r in results]

    avg_ecg_prob = np.mean(ecg_probs)
    avg_risk = np.mean(risk_scores)
    max_risk = max(risk_scores)
    min_risk = min(risk_scores)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average ECG Abnormality", f"{avg_ecg_prob:.2f}")
    col2.metric("Average Risk Score", f"{avg_risk:.2%}")
    col3.metric("Max Risk", f"{max_risk:.2%}")
    col4.metric("Min Risk", f"{min_risk:.2%}")

    st.divider()
    st.subheader("Prediction History")

    results_sorted = sorted(results, key=lambda x: x["timestamp"], reverse=True)

    for r in results_sorted:
        header = (
            f"{r['timestamp'][:19]} | "
            f"Risk: {r['cardio_risk_score']:.2%} | "
            f"ECG: {r['ecg_prediction']}"
        )

        with st.expander(header):
            features = r.get("features", {})

            st.markdown("**Demographics & Inputs**")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Age", features.get("age"))
                st.metric("Sex", features.get("sex"))
                st.metric("BMI", round(features.get("bmi", 0), 1))

            with c2:
                st.metric("Exercise Quality", features.get("exercise"))
                st.metric("Sleep Quality", features.get("sleep"))
                st.metric("Diet Quality", features.get("diet"))

            with c3:
                st.metric("Smoking / week", features.get("smoking_per_week"))
                st.metric("Alcohol / week", features.get("alcohol_per_week"))
                st.metric(
                    "Blood Pressure",
                    f"{features.get('sys_bp')}/{features.get('dia_bp')}"
                )

            st.divider()

            st.markdown("**Model Outputs**")
            c1, c2, c3 = st.columns(3)
            c1.metric("ECG Abnormality Prob", f"{r['ecg_abnormality_prob']:.2f}")
            c2.metric("Risk Score", f"{r['cardio_risk_score']:.2%}")
            c3.metric("Classification", r["ecg_prediction"])

            st.markdown("**Findings**")

            findings = generate_findings(r, avg_risk)
            for text, severity in findings:
                if severity == 0:
                    st.success(text)
                elif severity == 1:
                    st.info(text)
                elif severity == 2:
                    st.warning(text)
                else:
                    st.error(text)
    st.divider()
    st.subheader("Trends Over Time")

    timestamps = [pd.to_datetime(r["timestamp"]) for r in results_sorted]
    risk_scores = [r["cardio_risk_score"] for r in results_sorted]
    ecg_probs = [r["ecg_abnormality_prob"] for r in results_sorted]

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode="lines+markers",
        line=dict(color="crimson", width=2),
        name="Risk",
        hovertemplate="Date: %{x|%b %d}<br>Risk: %{y:.2%}<extra></extra>"
    ))
    fig_risk.add_hline(y=avg_risk, line=dict(color="gold", dash="dot"), annotation_text="Personal Avg", annotation_position="top right")
    fig_risk.update_layout(title="Risk Score Over Time", yaxis=dict(range=[0, 1], title="Risk Score"), height=300)
    st.plotly_chart(fig_risk, use_container_width=True)

    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(
        x=timestamps,
        y=ecg_probs,
        mode="lines+markers",
        line=dict(color="darkblue", width=2),
        name="ECG Prob",
        hovertemplate="Date: %{x|%b %d}<br>ECG Prob: %{y:.2f}<extra></extra>"
    ))

    fig_ecg.add_hline(y=avg_ecg_prob, line=dict(color="gold", dash="dot"), annotation_text="Personal Avg", annotation_position="top right")
    fig_ecg.update_layout(title="ECG Abnormality Probability Over Time", yaxis=dict(range=[0, 1], title="Probability"), height=300)
    st.plotly_chart(fig_ecg, use_container_width=True)

    st.divider()
    st.subheader("Download History")
    if st.button("Prepare Full PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf_file = generate_profile_pdf(profile, results_sorted)
            st.download_button(
                "Download PDF",
                data=pdf_file,
                file_name="CardioIQ_Profile_Report.pdf",
                mime="application/pdf"
            )