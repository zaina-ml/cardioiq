import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import json
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

from cardioiq import ECGNet, TrainCfg
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

st.title("CardioIQ: ECG Abnormality Detection")

MODEL_PATH = "cardioiq_model.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Keep model local and ignored by git.")

    cfg = TrainCfg()
    model = ECGNet(input_len=cfg.crop_len)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    target_layer = model.r3.bn2
    gradcam = GradCAM(model, target_layer)

    return model, cfg, gradcam

try:
    model, cfg, gradcam = load_model()
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


def save_result(user_uid, user_email, filename, prediction, prob):
    try:
        doc_ref = db.collection("user_results").document(user_uid)
        doc_ref.set({
            "user_email": user_email,
            "results": firestore.ArrayUnion([{
                "filename": filename,
                "prediction": prediction,
                "probability": float(prob)
            }])
        }, merge=True)
    except Exception as e:
        st.error("Failed to save result.")

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
        st.experimental_rerun()


tabs = st.tabs(["Predict", "Profile", "About"])

with tabs[0]:
    if 'user' not in st.session_state:
        st.info("Please login or sign up to use the app.")
    else:
        uploaded_file = st.file_uploader("Upload single-lead ECG CSV", type=["csv"])
        if uploaded_file:
            try:
                ecg_data = np.loadtxt(uploaded_file, delimiter=",")
            except Exception:
                st.error("Invalid CSV file.")
                ecg_data = None

            if ecg_data is not None:
                ecg_data = (ecg_data - np.mean(ecg_data)) / (np.std(ecg_data) + 1e-8)

                if len(ecg_data) < cfg.crop_len:
                    ecg_data = np.pad(ecg_data, (0, cfg.crop_len - len(ecg_data)))
                else:
                    ecg_data = ecg_data[:cfg.crop_len]

                ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                with st.spinner("Analyzing ECG..."):
                    with torch.inference_mode():
                        prob = torch.sigmoid(model(ecg_tensor)).item()

                    cam, raw_output = gradcam(ecg_tensor)
                    cam = cam.squeeze()

                label = "Abnormal" if prob >= 0.88 else "Normal"

                st.markdown(
                    f"<p class='{ 'big-font' if label == 'Abnormal' else 'normal-font' }'>Prediction: {label}</p>",
                    unsafe_allow_html=True,
                )

                st.metric(label="Abnormal Probability", value=f"{prob:.2f}")
                st.progress(int(prob * 100))

                fig = plot_ecg_cam(ecg_tensor, cam, label)
                st.plotly_chart(fig, use_container_width=True)
                
                user_uid = st.session_state['user']['localId']
                user_email = st.session_state['user']['email']
                safe_filename = getattr(uploaded_file, "name", "uploaded_ecg.csv")
                save_result(user_uid, user_email, safe_filename, label, prob)

with tabs[1]:
    if 'user' not in st.session_state:
        st.info("Please login to view your profile.")
    else:
        st.header(f"Profile: {st.session_state['user']['email']}")
        user_uid = st.session_state['user']['localId']
        results = get_user_results(user_uid)
        if results:
            st.dataframe(results)
        else:
            st.info("No past results yet. Upload an ECG in the 'Predict' tab.")

with tabs[2]:
    st.header("About CardioIQ")
    st.write("""
    CardioIQ predicts abnormal heartbeats from single-lead ECGs using a neural network 
    trained on the MIT-BIH Arrhythmia Database.

    Features:
    - Advanced Deep learning model
    - Interactive ECG visualization with GRAD-CAM
    - User authentication with Firebase
    """)
