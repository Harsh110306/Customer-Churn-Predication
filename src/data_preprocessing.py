import os
import pandas as pd
import numpy as np
from src.utils import setup_logger, ensure_directories, DataPreprocessingError

logger = setup_logger("data_preprocessing")

class ChurnDataPreprocessor:
    """Class to handle data ingestion, cleaning, and preprocessing of the raw dataset."""

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_data(self) -> pd.DataFrame:
        """Loads the raw customer churn data."""
        logger.info(f"Loading raw dataset from {self.raw_data_path}")
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")

        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise DataPreprocessingError(f"Failed to load raw data: {str(e)}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and processes the raw dataset."""
        logger.info("Starting data cleaning process...")
        try:
            df_clean = df.copy()

            cols_to_drop = [
                'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
                'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'Churn Reason'
            ]

            existing_cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
            df_clean.drop(columns=existing_cols_to_drop, inplace=True)
            logger.info(f"Dropped columns: {existing_cols_to_drop}. New shape: {df_clean.shape}")

            if 'Total Charges' in df_clean.columns:
                df_clean['Total Charges'] = pd.to_numeric(df_clean['Total Charges'].astype(str).str.strip(), errors='coerce')

                missing_total_charges = df_clean['Total Charges'].isna().sum()
                logger.info(f"Found {missing_total_charges} missing/blank values in Total Charges.")

                df_clean['Total Charges'] = df_clean['Total Charges'].fillna(0.0)
                logger.info("Filled missing values in Total Charges with 0.0")

            duplicates_count = df_clean.duplicated().sum()
            if duplicates_count > 0:
                df_clean.drop_duplicates(inplace=True)
                logger.info(f"Removed {duplicates_count} duplicate rows. New shape: {df_clean.shape}")

            if 'Churn Value' in df_clean.columns:
                churn_counts = df_clean['Churn Value'].value_counts()
                logger.info(f"Target distribution (Churn Value): \n{churn_counts}")
            else:
                logger.warning("Target variable 'Churn Value' not found in cleaned dataset.")

            return df_clean

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise DataPreprocessingError(f"Failed to clean data: {str(e)}")

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Saves the cleaned dataset to the processed data directory."""
        try:
            ensure_directories([os.path.dirname(self.processed_data_path)])
            df.to_csv(self.processed_data_path, index=False)
            logger.info(f"Cleaned dataset saved successfully to {self.processed_data_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise DataPreprocessingError(f"Failed to save processed data: {str(e)}")

    def run_pipeline(self) -> pd.DataFrame:
        """Runs the complete ingestion and cleaning pipeline."""
        df = self.load_data()
        df_clean = self.clean_data(df)
        self.save_processed_data(df_clean)
        return df_clean

if __name__ == "__main__":
    preprocessor = ChurnDataPreprocessor(
        raw_data_path="data/raw/Telco_customer_churn.csv",
        processed_data_path="data/processed/cleaned_churn.csv"
    )
    preprocessor.run_pipeline()
