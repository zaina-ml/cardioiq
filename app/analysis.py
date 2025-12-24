from config import THRESHOLDS

def generate_findings(result, avg_risk):
    findings = []
    f = result.get("features", {})

    ecg_prob = result.get("ecg_abnormality_prob", 0)
    risk = result.get("cardio_risk_score", 0)
    delta_risk = risk - avg_risk

    sys_bp = f.get("sys_bp")
    dia_bp = f.get("dia_bp")

    if ecg_prob >= THRESHOLDS["ECG_ABNORMAL_HIGH"]:
        findings.append(("ECG abnormality probability markedly elevated", 3))
    elif ecg_prob >= THRESHOLDS["ECG_ABNORMAL_MODERATE"]:
        findings.append(("ECG abnormality probability moderately elevated", 2))
    elif ecg_prob < THRESHOLDS["ECG_ABNORMAL_LOW"]:
        findings.append(("ECG abnormality probability low", 0))

    if risk >= THRESHOLDS["RISK_HIGH"]:
        findings.append(("Overall cardiovascular risk high", 3))
    elif risk >= THRESHOLDS["RISK_MODERATE"]:
        findings.append(("Overall cardiovascular risk moderately elevated", 2))
    elif risk < THRESHOLDS["RISK_LOW"]:
        findings.append(("Overall cardiovascular risk low", 0))

    if delta_risk >= THRESHOLDS["DELTA_RISK_HIGH"]:
        findings.append(("Risk score significantly above personal baseline", 3))
    elif delta_risk >= THRESHOLDS["DELTA_RISK_MODERATE"]:
        findings.append(("Risk score mildly above personal baseline", 2))
    elif delta_risk <= -THRESHOLDS["DELTA_RISK_HIGH"]:
        findings.append(("Risk score well below personal baseline", 0))

    if sys_bp is not None and dia_bp is not None:
        if sys_bp >= THRESHOLDS["BP_SYS_HIGH"] or dia_bp >= THRESHOLDS["BP_DIA_HIGH"]:
            findings.append(("Blood pressure in hypertensive range", 3))
        elif sys_bp >= THRESHOLDS["BP_SYS_ELEVATED"] or dia_bp >= THRESHOLDS["BP_DIA_ELEVATED"]:
            findings.append(("Blood pressure elevated", 2))
        elif sys_bp < 120 and dia_bp < 80:
            findings.append(("Blood pressure within normal range", 0))

    if f.get("exercise") is not None and f.get("exercise") < 0.3:
        findings.append(("Low physical activity", 2))
    if f.get("sleep") is not None and f.get("sleep") < 0.3:
        findings.append(("Sleep quality below recommended range", 2))
    if f.get("smoking_per_week", 0) > 0:
        findings.append(("Active tobacco exposure reported", 3))
    if f.get("alcohol_per_week", 0) > 14:
        findings.append(("High alcohol consumption reported", 2))

    if not findings:
        findings.append(("No values outside expected range detected", 0))

    return findings
