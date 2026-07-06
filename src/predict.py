import os
import pandas as pd
import numpy as np
import joblib
import shap
from src.utils import setup_logger, ModelInferenceError
from src.feature_engineering import ChurnFeatureEngineer

logger = setup_logger("predict")

class ChurnPredictor:
    """Class to handle end-to-end inference on raw customer data."""

    def __init__(self,
                 model_path: str = "models/churn_model.pkl",
                 encoder_path: str = "models/encoder.joblib",
                 scaler_path: str = "models/scaler.joblib",
                 metadata_path: str = "models/model_metadata.joblib",
                 background_path: str = "models/shap_background.joblib"):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.background_path = background_path

        self.model = None
        self.fe = None
        self.metadata = None
        self.explainer = None
        self.background_data = None

        self.defaults = {
            'Gender': 'Female',
            'Senior Citizen': 'No',
            'Partner': 'No',
            'Dependents': 'No',
            'Tenure Months': 1,
            'Phone Service': 'Yes',
            'Multiple Lines': 'No',
            'Internet Service': 'DSL',
            'Online Security': 'No',
            'Online Backup': 'No',
            'Device Protection': 'No',
            'Tech Support': 'No',
            'Streaming TV': 'No',
            'Streaming Movies': 'No',
            'Contract': 'Month-to-month',
            'Paperless Billing': 'Yes',
            'Payment Method': 'Mailed check',
            'Monthly Charges': 50.0,
            'Total Charges': 50.0,
            'CLTV': 3000
        }

        self.load_models()

    def load_models(self) -> None:
        """Loads model, feature engineer, and shap explainer from disk."""
        logger.info("Loading inference models and assets...")
        try:
            if not (os.path.exists(self.model_path) and os.path.exists(self.encoder_path) and os.path.exists(self.scaler_path)):
                raise FileNotFoundError("Model, encoder, or scaler asset is missing. Please run src/train.py first.")

            self.model = joblib.load(self.model_path)
            self.fe = ChurnFeatureEngineer(encoder_path=self.encoder_path, scaler_path=self.scaler_path)
            self.fe.load_assets()

            if os.path.exists(self.metadata_path):
                self.metadata = joblib.load(self.metadata_path)

            if os.path.exists(self.background_path):
                self.background_data = joblib.load(self.background_path)

            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as ex_shap:
                logger.warning(f"Could not load TreeExplainer directly, setting up fallback: {str(ex_shap)}")
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_data.sample(10, random_state=42))

            logger.info("All model assets loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model assets: {str(e)}")
            raise ModelInferenceError(f"Initialization failure: {str(e)}")

    def get_retention_recommendations(self, data: dict, churn_prob: float) -> list:
        """Generates contextual retention recommendations based on customer attributes and risk."""
        recs = []
        if churn_prob < 0.3:
            recs.append("Customer is low risk. Maintain standard relationship management.")
            return recs

        monthly_charges = float(data.get('Monthly Charges', 0))
        if monthly_charges > 80.0:
            recs.append("High monthly spend. Offer a loyalty discount, family plan, or recommend a cost-effective bundle.")

        contract = data.get('Contract', 'Month-to-month')
        if contract == 'Month-to-month':
            recs.append("Month-to-month contract. Suggest migrating to a 1-year or 2-year contract with a promotional 15% discount.")

        tech_support = data.get('Tech Support', 'No')
        if tech_support == 'No':
            recs.append("No technical support subscription. Offer a complimentary 3-month trial of the premium Tech Support package.")

        security = data.get('Online Security', 'No')
        backup = data.get('Online Backup', 'No')
        if security == 'No' or backup == 'No':
            recs.append("Lacks backup or security services. Recommend the 'Safe Home' digital package (Security + Backup) at a 20% bundle discount.")

        internet = data.get('Internet Service', 'DSL')
        if internet == 'Fiber optic':
            recs.append("Using Fiber Optic internet. Schedule a preemptive line check or customer satisfaction call to verify connection stability.")

        if not recs:
            recs.append("High churn score detected. Assign a customer service representative to conduct a proactive check-in call.")

        return recs

    def predict(self, customer_data: dict) -> dict:
        """Executes full preprocessing and prediction pipeline on a single customer dictionary."""
        try:
            full_data = self.defaults.copy()
            for key, val in customer_data.items():
                if val is not None and str(val).strip() != "":
                    if key in ['Tenure Months', 'CLTV']:
                        full_data[key] = int(val)
                    elif key in ['Monthly Charges', 'Total Charges']:
                        full_data[key] = float(val)
                    else:
                        full_data[key] = str(val)

            df_single = pd.DataFrame([full_data])

            df_engineered = self.fe.create_features(df_single)

            df_transformed = self.fe.transform(df_engineered)

            pred_label = int(self.model.predict(df_transformed)[0])
            pred_prob = float(self.model.predict_proba(df_transformed)[0, 1])

            risk_score = round(pred_prob * 100, 1)
            if pred_prob < 0.3:
                risk_level = "Low"
            elif pred_prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            recs = self.get_retention_recommendations(full_data, pred_prob)

            shap_explanations = []
            if self.explainer is not None:
                try:
                    raw_shap_values = self.explainer.shap_values(df_transformed)
                    if isinstance(raw_shap_values, list):
                        cust_shap = raw_shap_values[1][0]
                    else:
                        cust_shap = raw_shap_values[0]

                    if len(cust_shap.shape) == 2:
                        cust_shap = cust_shap[:, 1] if cust_shap.shape[1] == 2 else cust_shap[0]

                    feat_names = list(df_transformed.columns)
                    shap_map = list(zip(feat_names, cust_shap))

                    risk_drivers = sorted([item for item in shap_map if item[1] > 0.01], key=lambda x: x[1], reverse=True)[:4]
                    safety_factors = sorted([item for item in shap_map if item[1] < -0.01], key=lambda x: x[1])[:4]

                    def make_readable_name(feat_name):
                        if '_' in feat_name:
                            parts = feat_name.split('_')
                            return f"{parts[0]}: {parts[1]}"
                        return feat_name

                    shap_explanations = {
                        "risk_drivers": [{"feature": make_readable_name(f), "value": float(v)} for f, v in risk_drivers],
                        "safety_factors": [{"feature": make_readable_name(f), "value": float(v)} for f, v in safety_factors]
                    }
                except Exception as ex_shap_calc:
                    logger.warning(f"Failed to calculate local SHAP explanation: {str(ex_shap_calc)}")
                    shap_explanations = {"risk_drivers": [], "safety_factors": []}

            return {
                "prediction": "Churn" if pred_label == 1 else "No Churn",
                "churn_label": pred_label,
                "probability": pred_prob,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "recommendations": recs,
                "shap_explanation": shap_explanations
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ModelInferenceError(f"Failed to process prediction request: {str(e)}")

if __name__ == "__main__":
    try:
        predictor = ChurnPredictor()
        test_cust = {
            "Gender": "Male",
            "Tenure Months": 12,
            "Internet Service": "Fiber optic",
            "Contract": "Month-to-month",
            "Monthly Charges": 90.0,
            "Tech Support": "No",
            "Online Security": "No"
        }
        res = predictor.predict(test_cust)
        print("Prediction result:", res)
    except Exception as ex:
        print("Error during predict.py main execution:", str(ex))
