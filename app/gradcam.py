import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max(dim=-1, keepdim=True)[0] + 1e-6)
        return cam.cpu().numpy(), output.detach().cpu().numpy()


def plot_ecg_cam(ecg, cam, label, title="Uploaded ECG Signal", smooth_sigma=2, threshold=0.1):

    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy().squeeze()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy().squeeze()

    cam = gaussian_filter1d(cam, sigma=smooth_sigma)
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    if threshold is not None:
        cam[cam < threshold] = 0

    if label == "Abnormal":
        base_color = (255, 0, 0)
    else:
        base_color = (0, 200, 0) 

    colors = [
        f"rgba({base_color[0]},{base_color[1]},{base_color[2]},{a:.2f})"
        for a in cam
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=ecg,
        x=np.arange(len(ecg)),
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(color=colors, size=6),
        name='ECG'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        template="plotly_white",
        showlegend=False
    )

    return fig
