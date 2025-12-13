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
st.title("CardioIQ: AI-Powered ECG Analysis and Cardiovascular Risk Prediction")

ECG_MODEL_PATH = "ecgnet_model.pt"
CARDIORISK_MODEL_PATH = "cardiorisknet_model.pt"
ECG_WGAN_PATH = "ecg_cond_wgan_generator.pt"

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

def save_result(user_uid, user_email, filename, ecg_prob, risk_score):
    try:
        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        doc_ref = db.collection("user_results").document(user_uid)
        data = {
            "id": entry_id,
            "timestamp": timestamp,
            "user_email": user_email,
            "filename": filename,
            "ecg_abnormality_prob": float(ecg_prob),
            "ecg_prediction": "Abnormal" if ecg_prob >= 0.88 else "Normal",
            "cardio_risk_score": float(risk_score),
        }
        doc_ref.set({
            "user_email": user_email,
            "results": firestore.ArrayUnion([data])
        }, merge=True)
    except Exception as e:
        st.error(f"Failed to save result: {e}")

def get_user_results(user_uid):
    try:
        doc_ref = db.collection("user_results").document(user_uid)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("results", [])
    except Exception:
        pass
    return []


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
    if st.sidebar.button("Logout"): logout(); st.rerun()

tabs = st.tabs(["Predict", "Profile", "About"])

with tabs[0]:
    if 'user' not in st.session_state:
        st.info("Please login or sign up to use the app.")
    else:
        col_left, col_right = st.columns(2)
        window_size = 720

        # --- CSV Upload ---
        with col_left:
            st.write("### Import ECG CSV")
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                if ('uploaded_filename' not in st.session_state or
                        st.session_state['uploaded_filename'] != uploaded_file.name):
                    ecg_data = np.loadtxt(uploaded_file, delimiter=",")
                    ecg_data = (ecg_data - ecg_data.mean()) / (ecg_data.std() + 1e-8)
                    ecg_data = np.pad(ecg_data, (0, max(0, window_size - len(ecg_data))))[:window_size]
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
            fs = st.selectbox("Sampling rate", [250, 500, 1000], index=0)
            hr = st.slider("Heart rate (bpm)", 40, 160, 70)
            noise = st.slider("Noise", 0.0, 0.1, 0.01, step=0.005)

            disable_generate = st.session_state.get('source_label') == "CSV"
            generate = st.button("Generate ECG", disabled=disable_generate)
            if disable_generate:
                st.caption("Cannot generate ECG because a CSV is uploaded.")
            if generate:
                with torch.inference_mode():
                    ecg_wgan_model.eval()
                    z = torch.randn(1, 100)
                    y = torch.tensor([0])
                    ecg_fake = ecg_wgan_model(z, y).detach().cpu().numpy().squeeze()

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
            st.subheader("ECGNet Prediction")
            st.markdown(
                f"<p class='{ 'big-font' if st.session_state.ecg_label=='Abnormal' else 'normal-font' }'>Prediction: {st.session_state.ecg_label}</p>",
                unsafe_allow_html=True
            )
            st.metric(label="Abnormal Probability", value=f"{st.session_state.ecg_prob:.2f}")

            fig_cam = plot_ecg_cam(st.session_state['ecg_tensor'], st.session_state['ecg_cam'], st.session_state['ecg_label'])
            st.plotly_chart(fig_cam, use_container_width=True)

        st.subheader("Cardiovascular Risk Prediction")
        st.markdown("*(Only computed after ECG analysis)*")
        exercise = st.slider("Exercise Level (0=Sedentary, 1=Active)", 0.0, 1.0, 0.5)
        diet = st.slider("Diet Quality (0=Poor, 1=Ideal)", 0.0, 1.0, 0.5)
        sleep = st.slider("Sleep Quality (0=Poor, 1=Ideal)", 0.0, 1.0, 0.5)
        smoking = st.checkbox("Smoking")
        alcohol = st.checkbox("Alcohol Use")
        age = st.number_input("Age (years)", min_value=20, max_value=80, value=40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=25.0)
        bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=180, value=120)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=150, max_value=300, value=200)

        disable_risk = st.session_state.get('ecg_prob') is None
        if st.button("Compute Cardiovascular Risk", disabled=disable_risk):
            with st.spinner("Computing cardiovascular risk..."):
                ecg_prob = st.session_state['ecg_prob']
                age_norm = (age - 20)/60
                bmi_norm = (bmi - 18.5)/11.5
                bp_norm = (bp - 90)/90
                chol_norm = (chol - 150)/150

                risk_features = torch.tensor([
                    ecg_prob,
                    exercise, diet, sleep,
                    int(smoking), int(alcohol),
                    age_norm,
                    1 if sex=="Female" else 0,
                    bmi_norm, bp_norm, chol_norm
                ], dtype=torch.float32).unsqueeze(0)

                with torch.inference_mode():
                    risk_score = float(cardiorisk_model(risk_features).item())
                    st.session_state['risk_score'] = risk_score

            st.metric("Predicted Cardiovascular Risk", f"{st.session_state['risk_score']:.2%}")

            user_uid = st.session_state['user']['localId']
            user_email = st.session_state['user']['email']
            save_result(user_uid, user_email, st.session_state['source_label'], ecg_prob, risk_score)

with tabs[1]:
    if 'user' not in st.session_state:
        st.info("Please login to view your profile.")
        st.stop()

    user_uid = st.session_state['user']['localId']
    results = get_user_results(user_uid)

    if results:
        display_data = [{
            "Timestamp": r.get("timestamp", "-"),
            "Filename": r.get("filename", "-"),
            "ECG Prediction": r.get("ecg_prediction", "-"),
            "ECG Prob": f"{r.get('ecg_abnormality_prob', float('nan')):.2f}",
            "CardioRisk Score": f"{r.get('cardio_risk_score', float('nan')):.2%}"
        } for r in results]

        df = pd.DataFrame(display_data).sort_values("Timestamp", ascending=False)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No past results yet. Upload an ECG / Compute risk in 'Predict' tab.")

with tabs[2]:
    st.header("About CardioIQ")
    st.write("""
    CardioIQ is an AI-powered cardiovascular analysis tool that combines deep learning 
    ECG interpretation with a multi-factor cardiovascular risk model.

    Features:
    • ECGNet trained on MIT-BIH Arrhythmia Database  
    • Grad-CAM visualization for ECG  
    • Multi-factor risk prediction  
    • Secure Firebase authentication and storage
    """)
    st.write("zain aboobacker. 2025")
