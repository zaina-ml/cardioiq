import os
from utils import evaluate_model, compute_metrics, plot_pr_roc, plot_confusion, threshold_sweep
from model import ECGNet
from config import CFG

cfg = CFG
device = cfg.device

checkpoint_path = os.path.join("models", "cardioiq_model_modular.pt")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Model checkpoint not found at {checkpoint_path}. "
    )

checkpoint = torch.load(checkpoint_path, map_location=device)

model = ECGNet(input_len=cfg.crop_len, dropout_p=0.3)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

class EvalECGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X.squeeze(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.device = device
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        x, y = x.to(self.device), y.to(self.device)

        return x.unsqueeze(0), y

test_ds   = EvalECGDataset(X_test,   y_test, device=device)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

y_true, y_scores = evaluate_model(model, test_loader, device=device)

metrics, y_pred = compute_metrics(y_true, y_scores)

print(metrics)

plot_pr_roc(y_true, y_scores)

plot_confusion(y_true, y_pred)

best_t, best_f1 = threshold_sweep(y_true, y_scores)