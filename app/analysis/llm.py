
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": results_text + "\nUser question: " + user_input}
    ]

    api_url = "http://127.0.0.1:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio"
    }

    payload = {
        "model": "liquid/lfm2-1.2b",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 700
    }

    full_response = ""

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        full_response = data['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        full_response = f"Error connecting to AI server: {e}"
    
    return full_response