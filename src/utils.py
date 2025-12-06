def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    y_true, y_scores = [], []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
        y_true.append(y.cpu().numpy())
        y_scores.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    return y_true, y_scores

def compute_metrics(y_true, y_scores, threshold=0.82):
    y_pred = (y_scores >= threshold).astype(int)
    metrics = {
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': (y_pred == y_true).mean(),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    return metrics, y_pred

def plot_pr_roc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(recall, precision, label=f'PR-AUC={pr_auc:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(fpr, tpr, label=f'ROC-AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    plt.show()

def plot_confusion(y_true, y_scores, threshold=0.82):
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,6))
    plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix (threshold={threshold})')
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i,j]}', ha='center', va='center', color='red')
    plt.xticks([0,1], ['Neg','Pos']); plt.yticks([0,1], ['Neg','Pos'])
    plt.show()

def threshold_sweep(y_true, y_scores):
    thresholds = np.linspace(0, 1, 50)
    f1s = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred))

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    plt.plot(thresholds, f1s, label='F1')
    plt.axvline(best_t, color='r', linestyle='--', label=f'Best: {best_t:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.title('F1 vs Threshold')
    plt.legend()
    plt.show()

    return best_t, best_f1
