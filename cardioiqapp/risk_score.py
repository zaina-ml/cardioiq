def calculate_cardiovascular_risk(
    hr, systolic_bp, diastolic_bp, spo2,
    smoking_freq, alcohol_per_week, diet_score,
    sleep_hours, stress_level, age, sex,
    family_history, ecg_abnormal,
    weight_kg, height_cm
):
    score = 0
    
    if hr < 60:
        score += 2
    elif hr <= 100:
        score += 0
    else:
        score += 3
    
    if systolic_bp < 120 and diastolic_bp < 80:
        score += 0
    elif 120 <= systolic_bp <= 129 and diastolic_bp < 80:
        score += 1
    elif 130 <= systolic_bp <= 139 or 80 <= diastolic_bp <= 89:
        score += 2
    elif 140 <= systolic_bp <= 179 or 90 <= diastolic_bp <= 119:
        score += 4
    elif systolic_bp >= 180 or diastolic_bp >= 120:
        score += 6

    if spo2 >= 95:
        score += 0
    elif 90 <= spo2 < 95:
        score += 2
    else:
        score += 4

    smoking_weights = [0, 1, 3, 5]
    score += smoking_weights[smoking_freq]

    if alcohol_per_week == 0:
        score += 0
    elif 1 <= alcohol_per_week <= 3:
        score += 1
    elif 4 <= alcohol_per_week <= 7:
        score += 2
    else:
        score += 3

    score += diet_score

    if 7 <= sleep_hours <= 9:
        score += 0
    elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
        score += 1
    else:
        score += 2

    score += stress_level

    if age < 45:
        score += 0
    elif 45 <= age < 55:
        score += 1
    elif 55 <= age < 65:
        score += 2
    elif 65 <= age < 75:
        score += 3
    else:
        score += 4

    score += sex

    if family_history == 1:
        score += 2

    if ecg_abnormal == 1:
        score += 6

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        score += 1
    elif 18.5 <= bmi < 25:
        score += 0
    elif 25 <= bmi < 30:
        score += 2
    elif 30 <= bmi < 35:
        score += 4
    elif 35 <= bmi < 40:
        score += 6
    else:
        score += 8


    scaled_score = (score / 52) * 100

    if scaled_score <= 12.5:
        risk = "Very Low Risk"
    elif scaled_score <= 25:
        risk = "Low Risk"
    elif scaled_score <= 37.5:
        risk = "Moderate Risk"
    elif scaled_score <= 50:
        risk = "High Risk"
    elif scaled_score <= 75:
        risk = "Very High Risk"
    else:
        risk = "Critical Risk"

    return scaled_score, risk

