import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Real-time ECG Demo", layout="wide")
st.title("Smooth Real-time ECG Simulation")

# ----- Simulated ECG signal -----
ECG_LENGTH = 2000
t = np.linspace(0, 20 * np.pi, ECG_LENGTH)
ecg_signal = np.sin(t) + 0.05 * np.random.randn(ECG_LENGTH)

WINDOW_SIZE = 200  # visible samples

# ----- Session state -----
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'fig' not in st.session_state:
    # Initialize FigureWidget once
    fig = go.FigureWidget()
    fig.add_scatter(y=[0]*WINDOW_SIZE, mode='lines', line=dict(color='red'))
    fig.update_layout(
        yaxis=dict(range=[-1.5, 1.5]),
        xaxis=dict(range=[0, WINDOW_SIZE]),
        template="plotly_dark",
        title="ECG Signal"
    )
    st.session_state.fig = fig
    st.session_state.plot_placeholder = st.empty()
    st.session_state.plot_placeholder.plotly_chart(fig, use_container_width=True)

# ----- Start/Stop Controls -----
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start")
with col2:
    stop_btn = st.button("Stop")

if start_btn:
    st.session_state.playing = True
if stop_btn:
    st.session_state.playing = False

# ----- Real-time update loop -----
if st.session_state.playing:
    for _ in range(50):  # run 50 updates per click to avoid freezing Streamlit
        idx = st.session_state.idx
        start = max(0, idx - WINDOW_SIZE)
        window = ecg_signal[start:idx] if idx > 0 else np.zeros(WINDOW_SIZE)
        st.session_state.fig.data[0].y = window
        st.session_state.idx += 1
        if st.session_state.idx >= ECG_LENGTH:
            st.session_state.idx = 0
        time.sleep(0.05)
        st.session_state.plot_placeholder.plotly_chart(st.session_state.fig, use_container_width=True)
