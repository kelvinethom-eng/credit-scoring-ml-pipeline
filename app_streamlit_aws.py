"""
app_streamlit.py — Streamlit UI untuk Credit Scoring classifier di SageMaker.

Membaca nama endpoint & region dari environment variable.
boto3 mengambil kredensial AWS dari:
  - instance profile EC2 (saat jalan di EC2 dengan LabInstanceProfile), ATAU
  - ~/.aws/credentials (saat jalan lokal).
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-endpoints")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(record: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [record]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="wide")
st.title("💳 Credit Score Prediction")
st.write("Masukkan data pelanggan untuk memprediksi status kredit via SageMaker.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demografi & Pendapatan")
    age = st.number_input("Age", 18.0, 100.0, 42.0, 1.0)
    occupation = st.selectbox("Occupation", [
        "Manager", "Engineer", "Doctor", "Lawyer", "Teacher", "Scientist",
        "Entrepreneur", "Developer", "Journalist", "Mechanic", "Accountant",
        "Consultant", "Sales", "Other"])
    annual_income = st.number_input("Annual Income ($)", 0.0, value=350000.0, step=10000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary ($)", 0.0, value=25000.0, step=1000.0)
    monthly_balance = st.number_input("Monthly Balance ($)", value=15000.0, step=1000.0)
    amount_invested = st.number_input("Amount Invested Monthly ($)", 0.0, value=5000.0, step=100.0)

with col2:
    st.subheader("🏦 Akun & Hutang")
    num_bank_accounts = st.number_input("Num Bank Accounts", 0.0, 20.0, 3.0, 1.0)
    num_credit_card = st.number_input("Num Credit Card", 0.0, 20.0, 2.0, 1.0)
    num_of_loan = st.number_input("Num of Loan", 0.0, value=2.0, step=1.0)
    outstanding_debt = st.number_input("Outstanding Debt ($)", 0.0, value=20000.0, step=1000.0)
    total_emi = st.number_input("Total EMI per month ($)", 0.0, value=5000.0, step=500.0)
    interest_rate = st.number_input("Interest Rate (%)", 1.0, 100.0, 12.5, 0.5)

with col3:
    st.subheader("📊 Sejarah Kredit & Pembayaran")
    credit_history_age = st.text_input(
        "Credit History Age", "20 Years and 0 Months",
        help="Format: 'X Years and Y Months'")
    credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"], index=1)
    delay_due_date = st.number_input("Delay from due date (hari)", 0.0, value=5.0, step=1.0)
    num_delayed_payment = st.number_input("Num of Delayed Payment", 0.0, value=2.0, step=1.0)
    credit_util = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, 32.0, 1.0)
    changed_credit_limit = st.number_input("Changed Credit Limit", 0.0, value=2000.0, step=100.0)
    num_inquiries = st.number_input("Num Credit Inquiries", 0.0, value=4.0, step=1.0)
    payment_of_min_amount = st.selectbox("Payment of Min Amount", ["No", "Yes"], index=1)
    payment_behaviour = st.selectbox("Payment Behaviour", [
        "Low_spent_Last_Quarter", "High_spent_Last_Quarter",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
        "High_spent_Medium_value_payments", "High_spent_Large_value_payments"])

st.markdown("---")

if st.button("🔮 Prediksi Credit Score", type="primary", use_container_width=True):
    record = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Monthly_Balance": monthly_balance,
        "Amount_invested_monthly": amount_invested,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Num_of_Loan": num_of_loan,
        "Outstanding_Debt": outstanding_debt,
        "Total_EMI_per_month": total_emi,
        "Interest_Rate": interest_rate,
        "Credit_History_Age": credit_history_age,
        "Credit_Mix": credit_mix,
        "Delay_from_due_date": delay_due_date,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Credit_Utilization_Ratio": credit_util,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_inquiries,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Payment_Behaviour": payment_behaviour,
        "Month": "January",
    }

    try:
        result = invoke_endpoint(record)
    except NoCredentialsError:
        st.error(
            "Kredensial AWS tidak ditemukan. Di EC2: pasang LabInstanceProfile. "
            "Di lokal: konfigurasi ~/.aws/credentials.")
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]  # [P(Poor), P(Standard), P(Good)]

        if label == "Good":
            st.success(f"### ✅ {label.upper()} — profil kredit sehat")
        elif label == "Standard":
            st.info(f"### ⚠️ {label.upper()} — profil kredit wajar")
        else:
            st.error(f"### ❌ {label.upper()} — profil kredit berisiko")

        st.write("Probabilitas tiap kelas:")
        st.bar_chart({"probability": {
            "Poor": probs[0], "Standard": probs[1], "Good": probs[2]}})

with st.sidebar:
    st.markdown("## ℹ️ Info")
    st.json({"endpoint": ENDPOINT_NAME, "region": REGION})
