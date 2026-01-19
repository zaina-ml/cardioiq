MODEL_THRESHOLDS = {
    "THRESHOLD": 0.50,
}

THRESHOLDS = {
    "ECG_ABNORMAL_HIGH": 0.90,
    "ECG_ABNORMAL_MODERATE": 0.70,
    "ECG_ABNORMAL_LOW": 0.30,

    "RISK_HIGH": 0.70,
    "RISK_MODERATE": 0.50,
    "RISK_LOW": 0.30,

    "DELTA_RISK_HIGH": 0.15,
    "DELTA_RISK_MODERATE": 0.08,

    "BP_SYS_HIGH": 140,
    "BP_DIA_HIGH": 90,
    "BP_SYS_ELEVATED": 130,
    "BP_DIA_ELEVATED": 85,

    "EXERCISE_LOW": 0.3,
    "SLEEP_LOW": 0.3,
    "DIET_LOW": 0.3,

    "HR_HIGH": 100,
    "HR_LOW": 50,
    "SPO2_LOW": 92,

    "BMI_OBESE": 30,
    "BMI_OVERWEIGHT": 25,
    "BMI_UNDERWEIGHT": 18.5,
}

PDF_CONFIG = {
    "FONT": "Arial",
    "MARGIN": 15,
    "IMG_WIDTH": 180,
}


SYSTEM_PROMPT = """You are a cardiovascular AI assistant in CardioIQ. 
Analyze user data: Cardio Risk Score (0–100%), ECG Prediction (Normal/Abnormal, 0-1) and probability, Lifestyle (Exercise, Sleep, Diet 0–1; Smoking/Alcohol per week), and Demographics (Age, Sex, BMI, BP). 
Do not give medical diagnoses; always advise consulting a healthcare professional."""
