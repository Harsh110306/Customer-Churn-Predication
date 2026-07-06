import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import setup_logger, FeatureEngineeringError

logger = setup_logger("feature_engineering")

class ChurnFeatureEngineer:
    """Class to handle feature creation, encoding, and scaling."""

    def __init__(self, encoder_path: str = "models/encoder.joblib", scaler_path: str = "models/scaler.joblib"):
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.encoder = None
        self.scaler = None

        self.cat_cols = [
            'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
            'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
            'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
            'Contract', 'Paperless Billing', 'Payment Method', 'Tenure Group'
        ]

        self.num_cols = [
            'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV',
            'Average Monthly Spend', 'Customer Service Count'
        ]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new engineered features in the dataframe."""
        logger.info("Creating new features: Tenure Group, Average Monthly Spend, Customer Service Count")
        try:
            df_feat = df.copy()

            if 'Tenure Months' in df_feat.columns:
                def get_tenure_group(months):
                    if months <= 12:
                        return 'New Customer'
                    elif months <= 36:
                        return 'Medium Customer'
                    else:
                        return 'Loyal Customer'

                df_feat['Tenure Group'] = df_feat['Tenure Months'].apply(get_tenure_group)
            else:
                raise FeatureEngineeringError("'Tenure Months' is missing, cannot create 'Tenure Group'")

            if 'Total Charges' in df_feat.columns and 'Tenure Months' in df_feat.columns:
                df_feat['Average Monthly Spend'] = np.where(
                    df_feat['Tenure Months'] > 0,
                    df_feat['Total Charges'] / df_feat['Tenure Months'],
                    df_feat['Monthly Charges']
                )
            else:
                raise FeatureEngineeringError("'Total Charges' or 'Tenure Months' is missing, cannot create 'Average Monthly Spend'")

            service_cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
            available_service_cols = [col for col in service_cols if col in df_feat.columns]
            if available_service_cols:
                df_feat['Customer Service Count'] = df_feat[available_service_cols].apply(
                    lambda row: sum(1 for val in row if str(val).strip().lower() == 'yes'), axis=1
                )
            else:
                df_feat['Customer Service Count'] = 0
                logger.warning("No service columns found to compute Customer Service Count, defaulting to 0.")

            return df_feat
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise FeatureEngineeringError(f"Failed to create features: {str(e)}")

    def fit(self, X: pd.DataFrame) -> None:
        """Fits the OneHotEncoder and StandardScaler on the training data."""
        logger.info("Fitting encoder and scaler...")
        try:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(X[self.cat_cols])

            self.scaler = StandardScaler()
            self.scaler.fit(X[self.num_cols])

            logger.info("Successfully fitted encoder and scaler.")
        except Exception as e:
            logger.error(f"Error fitting encoder/scaler: {str(e)}")
            raise FeatureEngineeringError(f"Failed to fit preprocessing pipeline: {str(e)}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input data using the fitted encoder and scaler."""
        if self.encoder is None or self.scaler is None:
            raise FeatureEngineeringError("Encoder or Scaler is not fitted. Run fit() first or load saved assets.")

        try:
            X_temp = X.copy()

            cat_encoded = self.encoder.transform(X_temp[self.cat_cols])
            cat_encoded_cols = self.encoder.get_feature_names_out(self.cat_cols)
            df_cat = pd.DataFrame(cat_encoded, columns=cat_encoded_cols, index=X_temp.index)

            num_scaled = self.scaler.transform(X_temp[self.num_cols])
            df_num = pd.DataFrame(num_scaled, columns=self.num_cols, index=X_temp.index)

            X_transformed = pd.concat([df_num, df_cat], axis=1)
            logger.info(f"Transformed shape: {X_transformed.shape}")
            return X_transformed

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise FeatureEngineeringError(f"Failed to transform data: {str(e)}")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits encoder/scaler and transforms the training data."""
        self.fit(X)
        return self.transform(X)

    def save_assets(self) -> None:
        """Saves the fitted encoder and scaler to disk."""
        if self.encoder is None or self.scaler is None:
            raise FeatureEngineeringError("Cannot save unfitted encoder or scaler.")

        os.makedirs(os.path.dirname(self.encoder_path), exist_ok=True)
        joblib.dump(self.encoder, self.encoder_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Saved encoder to {self.encoder_path} and scaler to {self.scaler_path}")

    def load_assets(self) -> None:
        """Loads fitted encoder and scaler from disk."""
        if not os.path.exists(self.encoder_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Saved encoder or scaler files not found on disk.")

        self.encoder = joblib.load(self.encoder_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info("Loaded encoder and scaler assets successfully.")
