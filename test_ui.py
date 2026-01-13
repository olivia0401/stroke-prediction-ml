import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import StrokePredictor

predictor = StrokePredictor(artifact_path="models/model.pkl")

print("=" * 70)
print("中风预测系统测试")
print("=" * 70)

test_cases = [
    {
        "name": "高风险病例 (67岁男性, 心脏病, 高血糖, 高BMI)",
        "data": {
            "gender": "Male",
            "age": 67,
            "hypertension": 0,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Private",
            "residence_type": "Urban",
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "smoking_status": "formerly smoked"
        }
    },
    {
        "name": "低风险病例 (30岁女性, 健康指标正常)",
        "data": {
            "gender": "Female",
            "age": 30,
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "No",
            "work_type": "Private",
            "residence_type": "Urban",
            "avg_glucose_level": 85.0,
            "bmi": 22.5,
            "smoking_status": "never smoked"
        }
    },
    {
        "name": "中等风险病例 (45岁女性, 有高血压)",
        "data": {
            "gender": "Female",
            "age": 45,
            "hypertension": 1,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Self-employed",
            "residence_type": "Rural",
            "avg_glucose_level": 180.5,
            "bmi": 28.3,
            "smoking_status": "never smoked"
        }
    },
    {
        "name": "高风险病例 (80岁男性, 高血压+心脏病, 吸烟)",
        "data": {
            "gender": "Male",
            "age": 80,
            "hypertension": 1,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Private",
            "residence_type": "Urban",
            "avg_glucose_level": 200.0,
            "bmi": 30.0,
            "smoking_status": "smokes"
        }
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n测试 {i}: {test_case['name']}")
    print("-" * 70)

    input_df = pd.DataFrame([test_case['data']])

    prediction = predictor.predict(input_df)[0]
    proba = predictor.predict_proba(input_df)[0]

    risk_prob = float(proba[1]) * 100
    result = "高风险" if prediction == 1 else "低风险"

    print(f"预测结果: {result}")
    print(f"中风概率: {risk_prob:.2f}%")
    print(f"健康概率: {(100-risk_prob):.2f}%")

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)
