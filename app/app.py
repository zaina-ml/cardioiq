import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import json
from datetime import datetime
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

from ecgnet import ECGNet
from cardiorisknet import CardioRiskNet

from gradcam import GradCAM, plot_ecg_cam

firebase_config = dict(st.secrets["firebase"])
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

def init_firebase_admin():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(".streamlit/service_account.json")
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error("Failed to initialize Firebase Admin.")
        raise

db = init_firebase_admin()


st.set_page_config(
    page_title="CardioIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CardioIQ: AI-Powered ECG Analysis and Cardiovascular Risk Prediction")

ECG_MODEL_PATH = "ecgnet_model.pt"
CARDIORISK_MODEL_PATH = "cardiorisknet_model.pt"

@st.cache_resource
def load_ecg_model():
    if not os.path.exists(ECG_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {ECG_MODEL_PATH}. Keep model local and ignored by git.")

    model = ECGNet()
    state = torch.load(ECG_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    target_layer = model.r3.bn2
    gradcam = GradCAM(model, target_layer)

    return model, gradcam

def load_risk_model(model_path="cardiorisknet_model.pt"):
    import os, torch
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = CardioRiskNet(input_size=11)
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model, gradcam = load_ecg_model()
    cardiorisk_model = load_risk_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    raise

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
        entry_id = str(uuid.uuid4())  # unique record
        timestamp = datetime.utcnow().isoformat()

        doc_ref = db.collection("user_results").document(user_uid)

        data = {
            "id": entry_id,
            "timestamp": timestamp,
            "user_email": user_email,
            "filename": filename,

            # ECG data
            "ecg_abnormality_prob": float(ecg_prob),
            "ecg_prediction": "Abnormal" if ecg_prob >= 0.88 else "Normal",

            # Cardiovascular risk
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

st.markdown("""
<style>
.big-font {font-size:28px !important; color:#ff4b4b; font-weight:bold;}
.normal-font {font-size:28px !important; color:#00b300; font-weight:bold;}
.stButton>button {background-color:#ff4b4b; color:white;}
</style>
""", unsafe_allow_html=True)


if 'user' not in st.session_state:
    auth_mode = st.sidebar.radio("Select Option", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Login":
        if st.sidebar.button("Login"):
            success, msg = login(email, password)
            if success:
                st.success("Logged in!")
                st.rerun()
            else:
                st.error(f"Login Failed: {msg}")

    else:
        if st.sidebar.button("Sign Up"):
            success, msg = signup(email, password)
            if success:
                st.success("Account created and logged in!")
                st.rerun()
            else:
                st.error(f"Signup Failed: {msg}")
else:
    st.sidebar.success(f"Logged in as {st.session_state['user']['email']}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()


tabs = st.tabs(["Predict", "Profile", "About"])
import uuid

with tabs[0]:
    if 'user' not in st.session_state:
        st.info("Please login or sign up to use the app.")
    else:
        uploaded_file = st.file_uploader("Upload single-lead ECG CSV", type=["csv"])

        if uploaded_file:
            # Reset session state if new file
            if 'uploaded_filename' not in st.session_state or st.session_state['uploaded_filename'] != uploaded_file.name:
                st.session_state.update({
                    'uploaded_filename': uploaded_file.name,
                    'ecg_tensor': None,
                    'ecg_prob': None,
                    'ecg_cam': None,
                    'risk_score': None
                })

            # Preprocess ECG once
            if st.session_state['ecg_tensor'] is None:
                ecg_data = np.loadtxt(uploaded_file, delimiter=",")
                ecg_data = (ecg_data - np.mean(ecg_data)) / (np.std(ecg_data) + 1e-8)
                ecg_data = np.pad(ecg_data, (0, max(0, 720 - len(ecg_data))))[:720]
                st.session_state['ecg_tensor'] = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            ecg_tensor = st.session_state['ecg_tensor']

            # Compute ECGNet once per file
            if st.session_state['ecg_prob'] is None:
                with st.spinner("Analyzing ECG..."):
                    with torch.inference_mode():
                        st.session_state['ecg_prob'] = float(torch.sigmoid(model(ecg_tensor)).item())
                    cam, _ = gradcam(ecg_tensor)
                    st.session_state['ecg_cam'] = cam.squeeze()

            ecg_prob = st.session_state['ecg_prob']
            cam = st.session_state['ecg_cam']
            ecg_label = "Abnormal" if ecg_prob >= 0.88 else "Normal"

            # --- ECGNet Section ---
            st.subheader("ECGNet Prediction")
            st.markdown(
                f"<p class='{ 'big-font' if ecg_label == 'Abnormal' else 'normal-font' }'>Prediction: {ecg_label}</p>",
                unsafe_allow_html=True,
            )
            st.metric(label="Abnormal Probability", value=f"{ecg_prob:.2f}")
            fig = plot_ecg_cam(ecg_tensor, cam, ecg_label)
            st.plotly_chart(fig, use_container_width=True)

            # --- Cardiovascular Risk Section ---
            st.subheader("Cardiovascular Risk Prediction")
            st.markdown("*(Utilizes ECGNet output above)*")
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

            # Button computes risk and saves both ECG + Risk
            if st.button("Compute Cardiovascular Risk", key="compute_risk"):
                # Normalize numeric inputs
                age_norm = (age - 20)/60
                bmi_norm = (bmi - 18.5)/11.5
                bp_norm = (bp - 90)/90
                chol_norm = (chol - 150)/150

                risk_features = torch.tensor([
                    ecg_prob,
                    exercise, diet, sleep,
                    int(smoking), int(alcohol),
                    age_norm,
                    1 if sex == "Female" else 0,
                    bmi_norm, bp_norm, chol_norm
                ], dtype=torch.float32).unsqueeze(0)

                with torch.inference_mode():
                    risk_score = float(cardiorisk_model(risk_features).item())
                    st.session_state['risk_score'] = risk_score

                user_uid = st.session_state['user']['localId']
                user_email = st.session_state['user']['email']

                save_result(
                    user_uid,
                    user_email,
                    uploaded_file.name,
                    ecg_prob,
                    risk_score
                )


            if st.session_state.get('risk_score') is not None:
                st.metric("Predicted Cardiovascular Risk", f"{st.session_state['risk_score']:.2%}")

with tabs[1]:
    if 'user' not in st.session_state:
        st.info("Please login to view your profile.")
    else:
        st.header(f"Profile: {st.session_state['user']['email']}")
        user_uid = st.session_state['user']['localId']
        results = get_user_results(user_uid)

with tabs[1]:
    if 'user' not in st.session_state:
        st.info("Please login to view your profile.")
        st.stop()
        
    user_uid = st.session_state['user']['localId']
    results = get_user_results(user_uid)

    if results:
        display_data = []

        for entry in results:
            display_data.append({
                "Timestamp": entry.get("timestamp", "-"),
                "Filename": entry.get("filename", "-"),
                "ECG Prediction": entry.get("ecg_prediction", "-"),
                "ECG Prob": f"{entry.get('ecg_abnormality_prob', float('nan')):.2f}"
                    if entry.get("ecg_abnormality_prob") is not None else "-",
                "CardioRisk Score": f"{entry.get('cardio_risk_score', float('nan')):.2%}"
                    if entry.get("cardio_risk_score") is not None else "-"
            })

        df = pd.DataFrame(display_data)
        df = df.sort_values("Timestamp", ascending=False)

        st.dataframe(df, use_container_width=True)

    else:
        st.info("No past results yet. Upload an ECG / Compute Cardiovascular Risk in the 'Predict' tab.")


with tabs[2]:
    st.header("About CardioIQ")
    st.write("""
    CardioIQ is an AI-powered cardiovascular analysis tool that combines deep learning 
    ECG interpretation with a multi-factor cardiovascular risk model.

    The system uses:
    • A custom ECGNet model trained on the MIT-BIH Arrhythmia Database  
    • Grad-CAM visualization to highlight important ECG regions  
    • A risk-prediction network incorporating lifestyle, clinical, and ECG features  
    • Secure user authentication and result storage through Firebase  

    CardioIQ is built to demonstrate how AI can support early detection and 
    monitoring of cardiovascular health.
    """)

