import streamlit as st
import requests
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# STYLING & THEME
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5rem;
    }
    .stSubheader {
        color: #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.title("💳 Credit Score Prediction App")
st.markdown("""
    ### 🎯 Aplikasi Prediksi Skor Kredit Pelanggan
    
    Aplikasi ini menggunakan **Machine Learning** untuk memprediksi status kredit pelanggan.
    Masukkan data pelanggan di bawah ini dan klik **"Prediksi Credit Score"** untuk mendapatkan hasil.
    
    ---
""")

# ═══════════════════════════════════════════════════════════════════════════
# API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

API_URL = "http://127.0.0.1:8000/predict"
API_HEALTH_URL = "http://127.0.0.1:8000/health"

# Check API status
@st.cache_resource
def check_api_status():
    try:
        response = requests.get(API_HEALTH_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

api_available = check_api_status()

# Display API status
if api_available:
    st.success("✅ API Status: Connected")
else:
    st.error("""
    ❌ API Status: Disconnected
    
    **Solusi:**
    1. Buka terminal baru
    2. Jalankan: `python api.py`
    3. Tunggu sampai muncul "Uvicorn running on http://0.0.0.0:8000"
    4. Refresh halaman ini
    """)

# ═══════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("## 📋 Input Data Pelanggan")

with st.form("credit_form", clear_on_submit=False):
    
    # ─────────────────────────────────────────────────────────────
    # COLUMN 1: Demografi & Pendapatan
    # ─────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demografi & Pendapatan")
        
        age = st.number_input(
            "Age (Umur)",
            min_value=18.0,
            max_value=100.0,
            value=42.0,
            step=1.0
        )
        
        occupation = st.selectbox(
            "Occupation (Pekerjaan)",
            ["Manager", "Engineer", "Doctor", "Lawyer", "Teacher", 
             "Scientist", "Entrepreneur", "Developer", "Journalist",
             "Mechanic", "Accountant", "Consultant", "Sales", "Other"]
        )
        
        annual_income = st.number_input(
            "Annual Income (Pendapatan Tahunan $)",
            min_value=0.0,
            value=350000.0,
            step=10000.0
        )
        
        monthly_salary = st.number_input(
            "Monthly Inhand Salary (Gaji Bulanan $)",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )
        
        monthly_balance = st.number_input(
            "Monthly Balance (Saldo Bulanan $)",
            value=15000.0,
            step=1000.0
        )
        
        amount_invested = st.number_input(
            "Amount Invested Monthly (Investasi Bulanan $)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )
    
    # ─────────────────────────────────────────────────────────────
    # COLUMN 2: Akun & Hutang
    # ─────────────────────────────────────────────────────────────
    with col2:
        st.subheader("🏦 Akun & Hutang")
        
        num_bank_accounts = st.number_input(
            "Num Bank Accounts (Jumlah Rekening Bank)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=1.0
        )
        
        num_credit_card = st.number_input(
            "Num Credit Card (Jumlah Kartu Kredit)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=1.0
        )
        
        num_of_loan = st.number_input(
            "Num of Loan (Jumlah Pinjaman)",
            min_value=0.0,
            value=2.0,
            step=1.0
        )
        
        outstanding_debt = st.number_input(
            "Outstanding Debt (Hutang Tertunda $)",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )
        
        total_emi = st.number_input(
            "Total EMI per month (Total EMI Bulanan $)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )
        
        interest_rate = st.number_input(
            "Interest Rate (Tingkat Bunga %)",
            min_value=1.0,
            max_value=100.0,
            value=12.5,
            step=0.5
        )
    
    # ─────────────────────────────────────────────────────────────
    # COLUMN 3: Sejarah Kredit & Pembayaran
    # ─────────────────────────────────────────────────────────────
    with col3:
        st.subheader("📊 Sejarah Kredit & Pembayaran")
        
        credit_history_age = st.text_input(
            "Credit History Age (Sejarah Kredit)",
            value="20 Years and 0 Months",
            help="Format: 'X Years and Y Months'"
        )
        
        credit_mix = st.selectbox(
            "Credit Mix (Campuran Kredit)",
            ["Bad", "Standard", "Good"],
            index=1
        )
        
        delay_due_date = st.number_input(
            "Delay from due date (Keterlambatan Hari)",
            min_value=0.0,
            value=5.0,
            step=1.0
        )
        
        num_delayed_payment = st.number_input(
            "Num of Delayed Payment (Jumlah Pembayaran Terlambat)",
            min_value=0.0,
            value=2.0,
            step=1.0
        )
        
        credit_util_ratio = st.number_input(
            "Credit Utilization Ratio (Rasio Penggunaan Kredit %)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=1.0
        )
        
        # Convert percentage to decimal (0-1 range)
        credit_util_ratio_decimal = credit_util_ratio
        
        changed_credit_limit = st.number_input(
            "Changed Credit Limit (Perubahan Limit Kredit)",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )
        
        num_inquiries = st.number_input(
            "Num Credit Inquiries (Jumlah Pertanyaan Kredit)",
            min_value=0.0,
            value=4.0,
            step=1.0
        )
        
        payment_of_min_amount = st.selectbox(
            "Payment of Min Amount (Pembayaran Jumlah Minimum)",
            ["No", "Yes"],
            index=1
        )
        
        payment_behaviour = st.selectbox(
            "Payment Behaviour (Perilaku Pembayaran)",
            ["Low_spent_Last_Quarter",
             "High_spent_Last_Quarter",
             "High_spent_Small_value_payments",
             "Low_spent_Large_value_payments",
             "Low_spent_Medium_value_payments",
             "Low_spent_Small_value_payments",
             "High_spent_Medium_value_payments",
             "High_spent_Large_value_payments"],
            index=0
        )
    
    # ─────────────────────────────────────────────────────────────
    # SUBMIT BUTTON
    # ─────────────────────────────────────────────────────────────
    st.markdown("---")
    submit_button = st.form_submit_button(
        "🔮 Prediksi Credit Score",
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════
# API REQUEST & RESPONSE HANDLING
# ═══════════════════════════════════════════════════════════════════════════

if submit_button:
    
    if not api_available:
        st.error("❌ API tidak tersedia. Silakan jalankan `python api.py` terlebih dahulu.")
    else:
        
        # ─────────────────────────────────────────────────────────────
        # PREPARE PAYLOAD
        # ─────────────────────────────────────────────────────────────
        
        payload = {
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
            "Credit_Utilization_Ratio": credit_util_ratio_decimal,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": num_inquiries,
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Payment_Behaviour": payment_behaviour,
            "Month": "January"
        }
        
        # ─────────────────────────────────────────────────────────────
        # SEND REQUEST TO API
        # ─────────────────────────────────────────────────────────────
        
        with st.spinner('⏳ Sedang memproses... Menghubungi API...'):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                
                # ─────────────────────────────────────────────────────
                # HANDLE RESPONSE
                # ─────────────────────────────────────────────────────
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Extract data dari response
                    status = result_data.get("status", "unknown")
                    prediction_class = result_data["prediction"]["class"]
                    confidence = result_data["prediction"]["confidence"]
                    probabilities = result_data.get("probabilities", {})
                    
                    # Display hasil
                    st.markdown("---")
                    st.markdown("## 📊 Hasil Prediksi")
                    
                    # Main prediction display
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        if prediction_class == 'Good':
                            st.success(f"""
                            ### ✅ {prediction_class.upper()}
                            
                            **Status:** Profil kredit sangat sehat
                            
                            **Rekomendasi:** 
                            - Setujui aplikasi
                            - Interest Rate: 8-10%
                            - Credit Limit: Tinggi
                            """)
                        
                        elif prediction_class == 'Standard':
                            st.info(f"""
                            ### ⚠️ {prediction_class.upper()}
                            
                            **Status:** Profil kredit dalam batas wajar
                            
                            **Rekomendasi:**
                            - Review lebih lanjut
                            - Interest Rate: 12-15%
                            - Credit Limit: Sedang
                            """)
                        
                        else:  # Poor
                            st.error(f"""
                            ### ❌ {prediction_class.upper()}
                            
                            **Status:** Profil kredit berisiko tinggi
                            
                            **Rekomendasi:**
                            - Perlu investigasi lebih dalam
                            - Interest Rate: 18-25%
                            - Credit Limit: Rendah/Ditolak
                            """)
                    
                    with col_result2:
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence*100:.2f}%",
                            delta="Model Accuracy"
                        )
                        
                        st.write("### 📈 Class Probabilities:")
                        
                        # Display probabilities
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        with col_p1:
                            st.metric(
                                label="🔴 Poor",
                                value=f"{probabilities.get('Poor', 0)*100:.1f}%"
                            )
                        
                        with col_p2:
                            st.metric(
                                label="🟡 Standard",
                                value=f"{probabilities.get('Standard', 0)*100:.1f}%"
                            )
                        
                        with col_p3:
                            st.metric(
                                label="🟢 Good",
                                value=f"{probabilities.get('Good', 0)*100:.1f}%"
                            )
                    
                    # Display confidence bar
                    st.write("### Confidence Distribution:")
                    
                    col_bar1, col_bar2, col_bar3 = st.columns(3)
                    
                    with col_bar1:
                        st.progress(
                            probabilities.get('Poor', 0),
                            text=f"Poor: {probabilities.get('Poor', 0)*100:.1f}%"
                        )
                    
                    with col_bar2:
                        st.progress(
                            probabilities.get('Standard', 0),
                            text=f"Standard: {probabilities.get('Standard', 0)*100:.1f}%"
                        )
                    
                    with col_bar3:
                        st.progress(
                            probabilities.get('Good', 0),
                            text=f"Good: {probabilities.get('Good', 0)*100:.1f}%"
                        )
                    
                    # Input summary
                    st.markdown("---")
                    st.markdown("### 📝 Input Summary")
                    
                    summary_data = {
                        "Age": f"{int(age)} years",
                        "Occupation": occupation,
                        "Annual Income": f"${annual_income:,.0f}",
                        "Monthly Balance": f"${monthly_balance:,.0f}",
                        "Credit Mix": credit_mix,
                        "Payment Behaviour": payment_behaviour,
                        "Number of Loans": f"{int(num_of_loan)}",
                        "Credit History": credit_history_age
                    }
                    
                    for key, value in summary_data.items():
                        st.write(f"**{key}:** {value}")
                
                else:
                    st.error(f"""
                    ❌ API Error (Status Code: {response.status_code})
                    
                    **Error Details:**
```json
                    {response.text}
```
                    """)
            
            except requests.exceptions.Timeout:
                st.error("""
                ❌ Request Timeout
                
                API tidak merespons dalam waktu yang diharapkan.
                Pastikan API server sedang berjalan dan tidak terganggu.
                """)
            
            except requests.exceptions.ConnectionError:
                st.error("""
                ❌ Connection Error - API Server Tidak Ditemukan
                
                **Solusi:**
                1. Buka terminal baru
                2. Jalankan: `python api.py`
                3. Tunggu sampai muncul "Uvicorn running on http://0.0.0.0:8000"
                4. Refresh halaman Streamlit ini (F5 atau Ctrl+R)
                
                **Pastikan:**
                - Python sudah terinstall
                - FastAPI dan Uvicorn sudah diinstall: `pip install fastapi uvicorn`
                - Model dan preprocessor ada di folder: `artifacts/`
                """)
            
            except Exception as e:
                st.error(f"""
                ❌ Unexpected Error
                
                **Error Message:**
                          {str(e)}
                
                Coba lagi atau hubungi administrator.
                """)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: INFO & DEBUG
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ℹ️ Informasi")
    
    st.info("""
    ### Cara Menggunakan:
    
    1. **Isi Form** - Masukkan data pelanggan
    2. **Klik Prediksi** - Klik tombol "Prediksi Credit Score"
    3. **Lihat Hasil** - Hasil prediksi akan ditampilkan
    
    ### Credit Score Classes:
    
    - **🟢 Good** - Profil kredit sangat sehat
    - **🟡 Standard** - Profil kredit wajar
    - **🔴 Poor** - Profil kredit berisiko
    """)
    
    st.markdown("---")
    
    st.markdown("## 🔧 Debug Info")
    
    if st.checkbox("Show API Status Details"):
        try:
            health_response = requests.get(API_HEALTH_URL, timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.json(health_data)
            else:
                st.error("API Health Check Failed")
        except Exception as e:
            st.error(f"Cannot connect to API: {str(e)}")
    