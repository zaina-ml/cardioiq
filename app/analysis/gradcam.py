import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from config import MODEL_THRESHOLDS

class GradCAM:
    def __init__(self, model, target_layer, abnormal_class_idx=1):
        self.model = model
        self.target_layer = target_layer
        self.class_idx = abnormal_class_idx
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)

        score = output.sum()
        score.backward()

        weights = self.gradients.mean(dim=-1, keepdim=True)  # [B, C, 1]
        cam = (weights * self.activations).sum(dim=1)        # [B, T]

        cam = cam / (cam.abs().max(dim=-1, keepdim=True)[0] + 1e-6)

        return cam.cpu().numpy(), output.detach().cpu().numpy()


def plot_ecg_cam(
    ecg,
    cam,
    pred_prob=None,
    title="ECG Abnormal Feature Attribution (Grad-CAM)",
    smooth_sigma=2,
    threshold=0.2
):
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy().squeeze()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy().squeeze()

    cam = gaussian_filter1d(cam, sigma=smooth_sigma)
    cam = np.clip(cam, 0, None)
    cam = cam / (cam.max() + 1e-6)

    if pred_prob is not None and pred_prob < MODEL_THRESHOLDS["THRESHOLD"]:
        cam[:] = 0

    colors = [
        f"rgba(255,0,0,{a:.2f})" if a > threshold else "rgba(0,0,0,0.25)"
        for a in cam
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(ecg)),
        y=ecg,
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(color=colors, size=6),
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        template="plotly_white"
    )

    return fig
