import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import random
from cardioiq import TrainCfg
from scipy.ndimage import gaussian_filter1d

def download_ecg_dataset(records=None, n_samples=20, lead=0, normalize=True, smooth_sigma=1):
    cfg = TrainCfg()
    fs = 360
    crop_len = cfg.crop_len
    window_sec = cfg.window_sec
    half = int(fs * window_sec / 2)

    if records is None:
        records = ["100","101","106","119","207","208","223"]

    os.makedirs("normal", exist_ok=True)
    os.makedirs("abnormal", exist_ok=True)

    def extract_beat(sig, center):
        if center < half or center + half > len(sig):
            return None
        seg = sig[center - half:center + half]
        if len(seg) < crop_len:
            seg = np.pad(seg, (0, crop_len - len(seg)))
        else:
            seg = seg[:crop_len]
        if normalize:
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        return seg

    np.random.seed(42)
    normals_all, abnormals_all = [], []

    for record in records:
        rec = wfdb.rdrecord(record, pn_dir="mitdb")
        ann = wfdb.rdann(record, "atr", pn_dir="mitdb")
        sig = rec.p_signal[:, lead]

        for s, sym in zip(ann.sample, ann.symbol):
            beat = extract_beat(sig, s)
            if beat is None:
                continue
            if sym == "N":
                normals_all.append((record, beat))
            elif sym in ("V", "F"):
                abnormals_all.append((record, beat))

    normals = random.sample(normals_all, k=min(n_samples, len(normals_all)))
    abnormals = random.sample(abnormals_all, k=min(n_samples, len(abnormals_all)))

    for i, (rec_name, seg) in enumerate(normals, 1):
        np.savetxt(f"normal/normal_{rec_name}_{i:03d}.csv", seg, delimiter=",")
    for i, (rec_name, seg) in enumerate(abnormals, 1):
        np.savetxt(f"abnormal/abnormal_{rec_name}_{i:03d}.csv", seg, delimiter=",")

    print(f"Saved {len(normals)} normal ECGs and {len(abnormals)} abnormal ECGs")
    print("Folders: 'normal/' and 'abnormal/'")

    t = np.linspace(0, window_sec, crop_len)
    plt.figure(figsize=(10,4))
    for _, seg in normals:
        y = gaussian_filter1d(seg, sigma=smooth_sigma)
        plt.plot(t, y, color="green", alpha=0.3)
    for _, seg in abnormals:
        y = gaussian_filter1d(seg, sigma=smooth_sigma)
        plt.plot(t, y, color="red", alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude" + (" (normalized)" if normalize else ""))
    plt.title("Overlay of Normal (green) vs Abnormal (red) ECG Beats")
    plt.tight_layout()
    plt.show()

    return len(normals), len(abnormals)

download_ecg_dataset(n_samples=20)
