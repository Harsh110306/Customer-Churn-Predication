import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logger, ChurnException
from src.predict import ChurnPredictor

logger = setup_logger("api")

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the predictor model on API startup."""
    global predictor
    logger.info("Starting up FastAPI application...")
    try:
        predictor = ChurnPredictor()
    except Exception as e:
        logger.error(f"Failed to load machine learning models during startup: {str(e)}")
        logger.warning("The model will need to be trained (python src/train.py) before the /predict endpoint can function.")
    yield

app = FastAPI(
    title="Customer Churn Prediction API",
    description="A production-grade REST API to predict telecom customer churn using demographic, service, and billing data.",
    version="1.0.0",
    lifespan=lifespan
)

class CustomerInput(BaseModel):
    Gender: Optional[str] = Field(default="Female", description="Gender of the customer (Male, Female)")
    Senior_Citizen: Optional[str] = Field(default="No", alias="Senior Citizen", description="Whether the customer is a senior citizen (Yes, No)")
    Partner: Optional[str] = Field(default="No", description="Whether the customer has a partner (Yes, No)")
    Dependents: Optional[str] = Field(default="No", description="Whether the customer has dependents (Yes, No)")
    Tenure_Months: Optional[int] = Field(default=1, alias="Tenure Months", description="Number of months the customer has stayed with the company", ge=0)
    Phone_Service: Optional[str] = Field(default="Yes", alias="Phone Service", description="Whether the customer has a phone service (Yes, No)")
    Multiple_Lines: Optional[str] = Field(default="No", alias="Multiple Lines", description="Whether the customer has multiple lines (Yes, No, No phone service)")
    Internet_Service: Optional[str] = Field(default="DSL", alias="Internet Service", description="Customer's internet service provider (DSL, Fiber optic, No)")
    Online_Security: Optional[str] = Field(default="No", alias="Online Security", description="Whether the customer has online security (Yes, No, No internet service)")
    Online_Backup: Optional[str] = Field(default="No", alias="Online Backup", description="Whether the customer has online backup (Yes, No, No internet service)")
    Device_Protection: Optional[str] = Field(default="No", alias="Device Protection", description="Whether the customer has device protection (Yes, No, No internet service)")
    Tech_Support: Optional[str] = Field(default="No", alias="Tech Support", description="Whether the customer has tech support (Yes, No, No internet service)")
    Streaming_TV: Optional[str] = Field(default="No", alias="Streaming TV", description="Whether the customer has streaming TV (Yes, No, No internet service)")
    Streaming_Movies: Optional[str] = Field(default="No", alias="Streaming Movies", description="Whether the customer has streaming movies (Yes, No, No internet service)")
    Contract: Optional[str] = Field(default="Month-to-month", description="The contract term of the customer (Month-to-month, One year, Two year)")
    Paperless_Billing: Optional[str] = Field(default="Yes", alias="Paperless Billing", description="Whether the customer has paperless billing (Yes, No)")
    Payment_Method: Optional[str] = Field(default="Mailed check", alias="Payment Method", description="The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))")
    Monthly_Charges: Optional[float] = Field(default=50.0, alias="Monthly Charges", description="The amount charged to the customer monthly", ge=0.0)
    Total_Charges: Optional[float] = Field(default=50.0, alias="Total Charges", description="The total amount charged to the customer", ge=0.0)
    CLTV: Optional[int] = Field(default=3000, description="Customer Lifetime Value", ge=0)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Gender": "Male",
                "Tenure Months": 12,
                "Internet Service": "Fiber optic",
                "Contract": "Month-to-month",
                "Monthly Charges": 90.0,
                "Total Charges": 1080.0,
                "CLTV": 3239,
                "Tech Support": "No",
                "Online Security": "No"
            }
        }
    )

class SHAPFeature(BaseModel):
    feature: str
    value: float

class SHAPExplanation(BaseModel):
    risk_drivers: List[SHAPFeature]
    safety_factors: List[SHAPFeature]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_score: float
    risk_level: str
    recommendations: List[str]
    shap_explanation: Optional[SHAPExplanation] = None

@app.get("/")
def read_root():
    """Root endpoint showing API status and model load state."""
    model_loaded = predictor is not None and predictor.model is not None
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "model_loaded": model_loaded,
        "model_metadata": predictor.metadata if model_loaded and predictor.metadata else None
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_churn(customer: CustomerInput):
    """Predicts churn probability, risk category, SHAP drivers, and offers recommendations for a customer."""
    global predictor

    if predictor is None or predictor.model is None:
        try:
            logger.info("Model not loaded yet. Attempting load...")
            predictor = ChurnPredictor()
        except Exception as e:
            logger.error(f"Inference called but models are missing: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model files are not available. Please run model training first."
            )

    try:
        customer_dict = customer.model_dump(by_alias=True)

        logger.info("Received prediction request")

        result = predictor.predict(customer_dict)

        return result

    except ChurnException as ce:
        logger.error(f"Business logic error during prediction: {str(ce)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ce)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
