def evaluate_model(model, dataset, cfg=CFG):
    model.eval()
    with torch.inference_mode():
        preds = model(dataset.features).squeeze()
        corr = torch.corrcoef(torch.stack([preds, dataset.targets]))[0,1]
    print(f"Correlation with target risk: {corr:.3f}")


def show_patient_metadata(features, true_risk, pred_risk):
    """
    features: 1D tensor of size 11
    true_risk: scalar tensor (0–1)
    pred_risk: scalar tensor (0–1)
    """

    (ecg_prob,
     exercise,
     diet,
     sleep,
     smoking,
     alcohol,
     age,
     sex,
     bmi,
     bp,
     chol) = features.tolist()

    print("\n================ Example Patient Metadata ================\n")

    print(f" ECG abnormality probability : {ecg_prob:.3f}")
    print(f" Exercise score (0–1)       : {exercise:.3f}")
    print(f" Diet quality (0–1)         : {diet:.3f}")
    print(f" Sleep quality (0–1)        : {sleep:.3f}")
    print(f" Smoking                    : {'Yes' if smoking > 0.5 else 'No'}")
    print(f" Alcohol use               : {'Yes' if alcohol > 0.5 else 'No'}")

    print(f" Age (normalized 0–1)       : {age:.3f}")
    print(f" Sex                        : {'Female' if sex > 0.5 else 'Male'}")
    print(f" BMI (normalized 0–1)       : {bmi:.3f}")
    print(f" Blood Pressure (0–1)       : {bp:.3f}")
    print(f" Cholesterol (0–1)          : {chol:.3f}")

    print("\n================ Risk Scores ================\n")
    print(f" Model Predicted Risk Score : {pred_risk:.3f}")
    print(f" True Risk Score            : {true_risk:.3f}")

    print("\n==================================================\n")

def analyze_dataset(dataset):
    """
    Computes and prints the average value of each input feature
    and the average target risk score.
    """
    features = dataset.features  # shape [N, 11]
    targets = dataset.targets    # shape [N]

    mean_feats = features.mean(dim=0)
    mean_target = targets.mean().item()

    (ecg_prob,
     exercise,
     diet,
     sleep,
     smoking,
     alcohol,
     age,
     sex,
     bmi,
     bp,
     chol) = mean_feats.tolist()

    print("\n================ Dataset Feature Averages ================\n")

    print(f" Avg ECG abnormality prob   : {ecg_prob:.3f}")
    print(f" Avg exercise score         : {exercise:.3f}")
    print(f" Avg diet score             : {diet:.3f}")
    print(f" Avg sleep score            : {sleep:.3f}")
    print(f" % Smokers                  : {smoking:.3f}")
    print(f" % Alcohol users            : {alcohol:.3f}")

    print(f" Avg age (normalized)       : {age:.3f}")
    print(f" % Female                   : {sex:.3f}")
    print(f" Avg BMI (normalized)       : {bmi:.3f}")
    print(f" Avg Blood Pressure (0-1)   : {bp:.3f}")
    print(f" Avg Cholesterol (0-1)      : {chol:.3f}")

    print("\n================ Label Averages ================\n")
    print(f" Avg Risk Score             : {mean_target:.3f}")

    print("\n===========================================================\n")
