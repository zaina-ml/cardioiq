import requests

def format_results(results, max_entries=5):
    results_sorted = sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    summary_lines = []
    for r in results_sorted[:max_entries]:
        features = r.get("features", {})

        line = (
            f"Date: {r['timestamp'][:19]}\n"
            f"Cardio Risk: {r['cardio_risk_score']:.2%}\n"
            f"ECG Prediction: {r['ecg_prediction']}\n"
            f"ECG Abnormality Probability: {r['ecg_abnormality_prob']:.2f}\n"
            f"Demographics & Inputs:\n"
            f"  - Age: {features.get('age', 'N/A')}\n"
            f"  - Sex: {features.get('sex', 'N/A')}\n"
            f"  - BMI: {round(features.get('bmi', 0), 1)}\n"
            f"  - Exercise Quality: {features.get('exercise', 'N/A')}\n"
            f"  - Sleep Quality: {features.get('sleep', 'N/A')}\n"
            f"  - Diet Quality: {features.get('diet', 'N/A')}\n"
            f"  - Smoking / week: {features.get('smoking_per_week', 'N/A')}\n"
            f"  - Alcohol / week: {features.get('alcohol_per_week', 'N/A')}\n"
            f"  - Blood Pressure: {features.get('sys_bp', 'N/A')}/{features.get('dia_bp', 'N/A')}\n"
            f"  - Heart Rate: {features.get('hr', 'N/A')}\n"
            f"  - SpO2: {features.get('spo2', 'N/A')}\n"
        )
        summary_lines.append(line)

    return "\n\n".join(summary_lines)

def generate_llm_response(user_input, results_text, SYSTEM_PROMPT):
    return (
        "The AI assistant is temporarily unavailable. "
        "Please check back later."
    )

    # Adding functionality to enable LLM response generation when the API is available.