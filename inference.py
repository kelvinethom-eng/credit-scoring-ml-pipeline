import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path


class CreditScorePredictionService:
    def __init__(self, model_path: str, preprocessor_path: str, label_classes_path: str = None):
        print("[INFERENCE] Loading model, preprocessor, dan label classes...")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded: {model_path}")

        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✅ Preprocessor loaded: {preprocessor_path}")

        # Load label classes (PENTING untuk mapping yang benar)
        if label_classes_path and Path(label_classes_path).exists():
            with open(label_classes_path, 'r') as f:
                self.label_classes = json.load(f)
            print(f"✅ Label classes loaded: {self.label_classes}")
        else:
            # Fallback ke default (harus sama dengan training)
            self.label_classes = ['Good', 'Poor', 'Standard']
            print(f"⚠️  Using default label classes: {self.label_classes}")

    def preprocess_input(self, input_data: dict) -> np.ndarray:
        # Convert dict ke DataFrame
        df = pd.DataFrame([input_data])

        # Preprocess menggunakan fitted preprocessor
        X_scaled = self.preprocessor.transform(df, include_target=False)

        return X_scaled

    def predict(self, input_data: dict) -> dict:
        # Preprocess
        X_scaled = self.preprocess_input(input_data)

        # Predict
        prediction_encoded = self.model.predict(X_scaled)[0]
        prediction = self.label_classes[int(prediction_encoded)]

        # Get confidence/probability jika model support
        confidence = None
        probabilities = None

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probs))
            # Map probabilities menggunakan label_classes
            probabilities = {
                self.label_classes[i]: float(probs[i]) for i in range(len(self.label_classes))
            }

        result = {
            'prediction': prediction,
            'prediction_label': int(prediction_encoded),
            'confidence': confidence,
            'probabilities': probabilities
        }

        return result


def create_test_cases():
    test_cases = {
        'poor': {
        "Age": 31,
        "Occupation": "Engineer",
        "Annual_Income": 32992,
        "Monthly_Inhand_Salary": 2717,
        "Monthly_Balance": 300,
        "Amount_invested_monthly": 121,
  "Num_Bank_Accounts": 7,
  "Num_Credit_Card": 7,
  "Num_of_Loan": 5,
  "Outstanding_Debt": 1942,
  "Total_EMI_per_month": 80,
  "Interest_Rate": 21,
  "Credit_History_Age": "14 Years and 4 Months",
  "Credit_Mix": "Bad",
  "Delay_from_due_date": 27,
  "Num_of_Delayed_Payment": 17,
  "Credit_Utilization_Ratio": 32,
  "Changed_Credit_Limit": 10,
  "Num_Credit_Inquiries": 8,
  "Payment_of_Min_Amount": "Yes",
  "Payment_Behaviour": "Low_spent_Small_value_payments",
  "Month": "January"
}
,
        'standard': {
            'Age': 42,
            'Occupation': 'Manager',
            'Annual_Income': 350000,
            'Monthly_Inhand_Salary': 25000,
            'Monthly_Balance': 15000,
            'Amount_invested_monthly': 5000,
            'Num_Bank_Accounts': 3,
            'Num_Credit_Card': 2,
            'Num_of_Loan': 2,
            'Outstanding_Debt': 20000,
            'Total_EMI_per_month': 5000,
            'Interest_Rate': 12.5,
            'Credit_History_Age': '20 Years and 0 Months',
            'Credit_Mix': 'Standard',
            'Delay_from_due_date': 5,
            'Num_of_Delayed_Payment': 2,
            'Credit_Utilization_Ratio': 45,
            'Changed_Credit_Limit': 2000,
            'Num_Credit_Inquiries': 4,
            'Payment_of_Min_Amount': 'Yes',
            'Payment_Behaviour': 'High_spent_Last_Quarter',
            'Month': 'January'
        },
        'good': {
  "Age": 36,
  "Occupation": "Teacher",
  "Annual_Income": 44891,
  "Monthly_Inhand_Salary": 3765,
  "Monthly_Balance": 400,
  "Amount_invested_monthly": 167,
  "Num_Bank_Accounts": 4,
  "Num_Credit_Card": 4,
  "Num_of_Loan": 2,
  "Outstanding_Debt": 754,
  "Total_EMI_per_month": 63,
  "Interest_Rate": 7,
  "Credit_History_Age": "31 Years and 3 Months",
  "Credit_Mix": "Good",
  "Delay_from_due_date": 10,
  "Num_of_Delayed_Payment": 8,
  "Credit_Utilization_Ratio": 33,
  "Changed_Credit_Limit": 7,
  "Num_Credit_Inquiries": 3,
  "Payment_of_Min_Amount": "No",
  "Payment_Behaviour": "High_spent_Medium_value_payments",
  "Month": "January"
}

    }

    return test_cases


if __name__ == "__main__":
    print("  TESTING INFERENCE SERVICE")

    # Load model, preprocessor, dan label classes
    model_path = "artifacts/best_model.pkl"
    preprocessor_path = "artifacts/preprocessor.pkl"
    label_classes_path = "artifacts/label_classes.json"

    service = CreditScorePredictionService(
        model_path, preprocessor_path, label_classes_path
    )

    # Get test cases
    test_cases = create_test_cases()

    # Run predictions
    print("\n" + "─"*70)
    print("TEST CASE RESULTS:")
    print("─"*70)

    for class_name, test_data in test_cases.items():
        print(f"\n🔹 Test Case: {class_name.upper()}")
        print(f"   Input features sample:")
        print(f"      Age: {test_data['Age']}")
        print(f"      Annual Income: {test_data['Annual_Income']}")
        print(f"      Credit Mix: {test_data['Credit_Mix']}")
        print(f"      Monthly Balance: {test_data['Monthly_Balance']}")

        result = service.predict(test_data)

        print(f"\n   📊 Prediction Result:")
        print(f"      Predicted Class: {result['prediction']}")
        print(f"      Confidence: {result['confidence']:.4f}" if result['confidence'] else "      Confidence: N/A")

        if result['probabilities']:
            print(f"      Probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"         {cls}: {prob:.4f}")

        print()