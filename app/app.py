import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import uuid
from datetime import datetime
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

from ecgnet import ECGNet
from cardiorisknet import CardioRiskNet
from ecg_wgan import ECGGenerator
from gradcam import GradCAM, plot_ecg_cam

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

ECG_MODEL_PATH = "ecgnet_model.pt"
CARDIORISK_MODEL_PATH = "cardiorisknet_model.pt"
ECG_WGAN_PATH = "ecg_wgan_generator.pt"


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
        "ecg_prediction": "Abnormal" if ecg_prob >= 0.88 else "Normal",
        "cardio_risk_score": float(risk_score),
        "features": features
    }

    db.collection("users") \
      .document(user_uid) \
      .collection("profiles") \
      .document(profile["id"]) \
      .collection("results") \
      .add(data)



if 'user' not in st.session_state:
    auth_mode = st.sidebar.radio("Select Option", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Login" and st.sidebar.button("Login"):
        success, msg = login(email, password)
        if success: st.success("Logged in!"); st.rerun()
        else: st.error(f"Login Failed: {msg}")
    elif auth_mode == "Sign Up" and st.sidebar.button("Sign Up"):
        success, msg = signup(email, password)
        if success: st.success("Account created and logged in!"); st.rerun()
        else: st.error(f"Signup Failed: {msg}")
else:
    st.sidebar.success(f"Logged in as {st.session_state['user']['email']}")
    
    user_uid = st.session_state["user"]["localId"]

    profiles_ref = (
        db.collection("users")
        .document(user_uid)
        .collection("profiles")
    )

    profiles = [p.to_dict() | {"id": p.id} for p in profiles_ref.stream()]
    
    if profiles:
        names = [p["name"] for p in profiles]
        selected = st.sidebar.selectbox("Active profile", names)

        if st.session_state.get("active_profile_name") != selected:
            st.session_state.active_profile_name = selected
            st.session_state.active_profile = next(p for p in profiles if p["name"] == selected)

            for key in ["ecg", "ecg_tensor", "ecg_prob", "ecg_cam", "risk_score",
                        "uploaded_filename", "source_label", "ecg_label", "fake_ecg"]:
                st.session_state.pop(key, None)

            st.toast("Switched to profile: " + selected)

    else:
        st.sidebar.info("No profiles yet")

    with st.sidebar.expander("Add new profile"):
        name = st.text_input("Name")
        age = st.number_input("Age", 0, 120, 40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        height_cm = st.number_input("Height (cm)", 100, 210, 170, step=1)
        weight_kg = st.number_input("Weight (kg)", 10.0, 200.0, 70.0, step=0.5)

        if st.button("Create profile"):
            profiles_ref.add({
                "name": name,
                "age": age,
                "sex": sex,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "created_at": datetime.utcnow().isoformat()
            })
            st.rerun()

    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

tabs = st.tabs(["Predict", "Profile", "About"])

with tabs[2]:
    st.header("About CardioIQ")
    st.write("""
    CardioIQ is an AI-powered cardiovascular analysis tool that combines deep learning 
    ECG interpretation with a multi-factor cardiovascular risk model.

    Features:\n
    • ECGNet trained on MIT-BIH Arrhythmia Database
    • Synthetic ECG generation using Conditional wGAN  
    • Grad-CAM visualization for ECG  
    • Multi-factor risk prediction using MLP 
    • Secure Firebase authentication and storage\n
    """)
    st.caption("zain aboobacker. 2025")

st.caption("This tool should not be used as a substitute for professional medical advice.")


with tabs[0]:
    if 'user' not in st.session_state:
        st.info("Please login or sign up to use the app.")
    else:
        profile = st.session_state.active_profile

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
            generate = st.button("Generate ECG", key="generate_ecg", use_container_width=True, disabled=disable_generate)
            if disable_generate:
                st.caption("Cannot generate ECG because a CSV is uploaded.")
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
                st.session_state['ecg_label'] = "Abnormal" if st.session_state['ecg_prob'] >= 0.88 else "Normal"

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

        exercise = st.slider("Exercise Level (0=Sedentary, 1=Active)", 0.0, 1.0, 0.5)
        diet = st.slider("Diet Quality (0=Poor, 1=Ideal)", 0.0, 1.0, 0.5)
        sleep = st.slider("Sleep Quality (0=Poor, 1=Ideal)", 0.0, 1.0, 0.5)

        smoking_per_week = st.slider(
            "Cigarettes per Week",
            min_value=0, max_value=140,value=0, step=1)

        alcohol_per_week = st.slider("Alcoholic Drinks per Week", min_value=0, max_value=30, value=0, step=1)

        age = profile["age"]
        sex = profile["sex"]

        height_cm = profile["height_cm"]
        weight_kg = profile["weight_kg"]

        sys_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=220, value=120)
        dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=200, value=80)

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

            def add(findings, condition, text, alert=1):
                if condition:
                    findings.append((text, alert))

            findings = []

            add(findings, r["ecg_abnormality_prob"] >= 0.85,
                "ECG abnormality probability markedly elevated", 1)

            add(findings, 0.70 <= r["ecg_abnormality_prob"] < 0.85,
                "ECG abnormality probability moderately elevated", 1)

            add(findings, r["ecg_abnormality_prob"] < 0.30,
                "ECG abnormality probability low", 0)

            add(findings, r["cardio_risk_score"] >= 0.70,
                "Overall cardiovascular risk high", 1)

            add(findings, 0.50 <= r["cardio_risk_score"] < 0.70,
                "Overall cardiovascular risk moderately elevated", 1)

            add(findings, r["cardio_risk_score"] < 0.30,
                "Overall cardiovascular risk low", 0)

            delta = r["cardio_risk_score"] - avg_risk

            add(findings, delta >= 0.15,
                "Risk score significantly above personal baseline", 1)

            add(findings, 0.08 <= delta < 0.15,
                "Risk score mildly above personal baseline", 1)

            add(findings, delta <= -0.15,
                "Risk score well below personal baseline", 0)

            sys_bp = features.get("sys_bp")
            dia_bp = features.get("dia_bp")

            if sys_bp and dia_bp:
                add(findings, sys_bp >= 140 or dia_bp >= 90,
                    "Blood pressure in hypertensive range", 1)

                add(findings, 130 <= sys_bp < 140 or 85 < dia_bp < 100,
                    "Blood pressure elevated", 1)

                add(findings, sys_bp < 120 and dia_bp < 80,
                    "Blood pressure within normal range", 0)

            add(findings, features.get("exercise", 0) <= 0.5,
                "Low physical activity", 1)

            add(findings, features.get("sleep", 0) <= 0.5,
                "Sleep quality below recommended range", 1)

            add(findings, features.get("smoking_per_week", 0) > 0,
                "Active tobacco exposure reported", 1)

            add(findings, features.get("alcohol_per_week", 0) > 14,
                "High alcohol consumption reported", 1)


            if not findings:
                findings.append(("No values outside expected range detected", 0))


            st.markdown("**Findings**")

            for f in findings:
                finding, alert = f
                if alert == 0:
                    st.success(finding)
                else:
                    st.error(finding)

    st.divider()
    st.subheader("Event Flags & Extremes")

    highest_risk = max(results, key=lambda x: x["cardio_risk_score"])
    most_abnormal = max(results, key=lambda x: x["ecg_abnormality_prob"])


    if len(risk_scores) >= 5:
        std = np.std(risk_scores)
        if std >= 0.12:
            st.warning("Risk scores show high variability over time")

        if std < 0.05:
            st.info("Risk scores stable over time")

    st.warning(
        f"Highest risk recorded: {highest_risk['cardio_risk_score']:.2%} "
        f"on {highest_risk['timestamp'][:10]}"
    )

    st.warning(
        f"Most abnormal ECG probability: {most_abnormal['ecg_abnormality_prob']:.2f} "
        f"on {most_abnormal['timestamp'][:10]}"
    )


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
        hovertemplate="Date: %{x|%b %d}<br>Risk: %{y:.2%}<extra></extra>"
    ))
    fig_risk.update_layout(title="Risk Score Over Time", yaxis=dict(range=[0, 1], title="Risk Score"), height=300)
    st.plotly_chart(fig_risk, use_container_width=True)

    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(
        x=timestamps,
        y=ecg_probs,
        mode="lines+markers",
        line=dict(color="darkblue", width=2),
        hovertemplate="Date: %{x|%b %d}<br>ECG Prob: %{y:.2f}<extra></extra>"
    ))
    fig_ecg.update_layout(title="ECG Abnormality Probability Over Time", yaxis=dict(range=[0, 1], title="Probability"), height=300)
    st.plotly_chart(fig_ecg, use_container_width=True)

