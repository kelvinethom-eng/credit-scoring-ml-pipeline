from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pandas as pd
import pickle
import uvicorn
from pathlib import Path
import logging
import json
from preprocessing import CreditDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

preprocessor = None
model = None
label_classes = None
MODEL_LOADED = False


def load_artifacts():
    global preprocessor, model, label_classes, MODEL_LOADED

    try:
        logger.info("📥 Loading artifacts...")

        # Load preprocessor
        preprocessor_path = 'artifacts/preprocessor.pkl'
        if not Path(preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        preprocessor = CreditDataPreprocessor.load(preprocessor_path)
        logger.info("✅ Preprocessor loaded successfully")

        # Load model
        model_path = 'artifacts/best_model.pkl'
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Model loaded successfully")

        # Load label_classes mapping (PENTING untuk konsistensi)
        label_classes_path = 'artifacts/label_classes.json'
        if Path(label_classes_path).exists():
            with open(label_classes_path, 'r') as f:
                label_classes = json.load(f)
            logger.info(f"✅ Label classes loaded: {label_classes}")
        else:
            logger.warning("⚠️ label_classes.json tidak ditemukan, menggunakan fallback mapping")
            label_classes = ['Good', 'Poor', 'Standard']

        MODEL_LOADED = True
        logger.info("✅ All artifacts loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {str(e)}")
        MODEL_LOADED = False

# ═══════════════════════════════════════════════════════════════════════════
# LIFESPAN CONTEXT (Pydantic v2)
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
    description="API untuk prediksi credit score customer menggunakan Machine Learning",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (Schema)
# ═══════════════════════════════════════════════════════════════════════════

class CreditInput(BaseModel):
    """Input schema untuk credit scoring"""

    Age: float = Field(..., gt=17, lt=101, description="Age of the customer (18-100)")
    Occupation: str = Field(..., description="Customer's occupation")
    Annual_Income: float = Field(..., ge=0, description="Annual income in dollars")
    Monthly_Inhand_Salary: float = Field(..., ge=0, description="Monthly salary in dollars")
    Monthly_Balance: float = Field(..., description="Monthly balance in dollars (can be negative)")
    Amount_invested_monthly: float = Field(..., ge=0, description="Monthly investment amount")
    Num_Bank_Accounts: float = Field(..., ge=0, le=20, description="Number of bank accounts")
    Num_Credit_Card: float = Field(..., ge=0, le=20, description="Number of credit cards")
    Num_of_Loan: float = Field(..., ge=0, description="Number of loans")
    Outstanding_Debt: float = Field(..., ge=0, description="Outstanding debt amount")
    Total_EMI_per_month: float = Field(..., ge=0, description="Total EMI per month")
    Interest_Rate: float = Field(..., ge=1, le=100, description="Interest rate (%)")
    Credit_History_Age: str = Field(..., description="Credit history age (e.g., '20 Years and 0 Months')")
    Credit_Mix: str = Field(..., description="Credit mix type: Bad, Standard, or Good")
    Delay_from_due_date: float = Field(..., ge=0, description="Days delayed from due date")
    Num_of_Delayed_Payment: float = Field(..., ge=0, description="Number of delayed payments")
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100, description="Credit utilization ratio    ")
    Changed_Credit_Limit: float = Field(..., ge=0, description="Changed credit limit")
    Num_Credit_Inquiries: float = Field(..., ge=0, description="Number of credit inquiries")
    Payment_of_Min_Amount: str = Field(..., description="Payment of minimum amount: Yes or No")
    Payment_Behaviour: str = Field(..., description="Payment behaviour type")
    Month: str = Field(default="January", description="Month")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 42,
                "Occupation": "Manager",
                "Annual_Income": 350000,
                "Monthly_Inhand_Salary": 25000,
                "Monthly_Balance": 15000,
                "Amount_invested_monthly": 5000,
                "Num_Bank_Accounts": 3,
                "Num_Credit_Card": 2,
                "Num_of_Loan": 2,
                "Outstanding_Debt": 20000,
                "Total_EMI_per_month": 5000,
                "Interest_Rate": 12.5,
                "Credit_History_Age": "7 Years and 0 Months",
                "Credit_Mix": "Standard",
                "Delay_from_due_date": 5,
                "Num_of_Delayed_Payment": 2,
                "Credit_Utilization_Ratio": 0.45,
                "Changed_Credit_Limit": 2000,
                "Num_Credit_Inquiries": 4,
                "Payment_of_Min_Amount": "Yes",
                "Payment_Behaviour": "High_spent_Last_Quarter",
                "Month": "January"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema untuk prediction response"""
    status: str
    prediction: dict
    probabilities: dict = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_ready: bool
    preprocessor_ready: bool
    api_version: str

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Root endpoint - Welcome message"""
    return {
        "message": "Welcome to Credit Scoring API",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return {
        "status": "✅ healthy" if MODEL_LOADED else "❌ unhealthy",
        "model_ready": model is not None,
        "preprocessor_ready": preprocessor is not None,
        "api_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_score(data: CreditInput):
    # Check if model is loaded
    if model is None or preprocessor is None:
        logger.error("❌ Model atau Preprocessor tidak loaded")
        raise HTTPException(
            status_code=503,
            detail="Model belum siap. Pastikan artifacts sudah di-load."
        )

    try:
        # Step 1: Convert Pydantic to dict
        input_dict = data.model_dump()
        logger.info(f"📥 Received input: {list(input_dict.keys())}")

        # Step 2: Convert dict to DataFrame
        input_df = pd.DataFrame([input_dict])
        logger.info(f"📥 Input shape: {input_df.shape}")

        # Step 3: Preprocessing
        X_processed = preprocessor.transform(input_df, include_target=False)
        logger.info(f"✅ Preprocessing successful | Shape: {X_processed.shape}")

        # Step 4: Model Inference
        prediction_encoded = model.predict(X_processed)[0]
        logger.info(f"🔮 Prediction encoded: {prediction_encoded}")

        # Step 5: Get probabilities
        probs_array = None
        confidence = None

        if hasattr(model, 'predict_proba'):
            probs_array = model.predict_proba(X_processed)[0]
            confidence = float(max(probs_array))
            logger.info(f"📈 Raw probs: {probs_array}")

        # Step 6: Map prediction to class name
        prediction_class = label_classes[int(prediction_encoded)]
        logger.info(f"✅ Prediction class: {prediction_class}")

        # Step 7: Return response
        response = {
            "status": "success",
            "prediction": {
                "class": prediction_class,
                "code": int(prediction_encoded),
                "confidence": confidence
            }
        }

        if probs_array is not None:
            response["probabilities"] = {
                label_classes[i]: float(probs_array[i]) for i in range(len(label_classes))
            }

        logger.info(f"✅ Response prepared: {response}")

        return response

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error saat pemrosesan: {str(e)}"
        )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - Run Server
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CREDIT SCORING API SERVER")
    print("="*70)
    print("\n📡 Server Configuration:")
    print(f"   Host: 0.0.0.0")
    print(f"   Port: 8000")
    print(f"\n🌐 Access Points:")
    print(f"   API Base: http://localhost:8000")
    print(f"   Swagger UI: http://localhost:8000/docs")
    print(f"   ReDoc: http://localhost:8000/redoc")
    print(f"   Health Check: http://localhost:8000/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )