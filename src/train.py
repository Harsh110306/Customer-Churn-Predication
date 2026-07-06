import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap

from src.utils import setup_logger, ensure_directories
from src.feature_engineering import ChurnFeatureEngineer

logger = setup_logger("train")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """Evaluates a model and returns key metrics."""
    logger.info(f"Evaluating {model_name}...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    logger.info(f"{model_name} Test Accuracy: {test_acc:.4f} | Recall (Churn): {recall:.4f} | F1: {f1:.4f}")

    return {
        "Model": model_name,
        "Train_Accuracy": train_acc,
        "Test_Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

def train_and_evaluate_all():
    """Main pipeline for training, comparing, tuning, and saving the model."""
    ensure_directories()

    cleaned_data_path = "data/processed/cleaned_churn.csv"
    if not os.path.exists(cleaned_data_path):
        logger.info("Cleaned dataset not found. Running preprocessor first...")
        from src.data_preprocessing import ChurnDataPreprocessor
        preprocessor = ChurnDataPreprocessor(
            raw_data_path="data/raw/Telco_customer_churn.csv",
            processed_data_path=cleaned_data_path
        )
        df = preprocessor.run_pipeline()
    else:
        df = pd.read_csv(cleaned_data_path)
        logger.info(f"Loaded processed dataset with shape: {df.shape}")

    if 'Churn Value' not in df.columns:
        raise ValueError("Target column 'Churn Value' not found in dataset.")

    X = df.drop(columns=['Churn Value'])
    y = df['Churn Value']

    fe = ChurnFeatureEngineer()
    X_engineered = fe.create_features(X)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train split shape: {X_train_raw.shape}, Test split shape: {X_test_raw.shape}")

    X_train = fe.fit_transform(X_train_raw)
    X_test = fe.transform(X_test_raw)
    fe.save_assets()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(metrics)

    df_results = pd.DataFrame(results).drop(columns=['y_pred', 'y_prob'])
    logger.info(f"\n=== Model Comparison Results ===\n{df_results.to_string(index=False)}")

    df_results.to_csv("data/processed/model_comparison.csv", index=False)

    logger.info("Starting hyperparameter tuning...")

    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    logger.info("Tuning Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    logger.info(f"RF Best Params: {rf_grid.best_params_}")

    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [1, 3]
    }
    logger.info("Tuning XGBoost...")
    xgb_grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        param_grid=xgb_param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    logger.info(f"XGBoost Best Params: {xgb_grid.best_params_}")

    rf_tuned_metrics = evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Tuned Random Forest")
    xgb_tuned_metrics = evaluate_model(best_xgb, X_train, y_train, X_test, y_test, "Tuned XGBoost")

    if xgb_tuned_metrics['F1_Score'] >= rf_tuned_metrics['F1_Score']:
        best_model_name = "Tuned XGBoost"
        best_model = best_xgb
        best_metrics = xgb_tuned_metrics
    else:
        best_model_name = "Tuned Random Forest"
        best_model = best_rf
        best_metrics = rf_tuned_metrics

    logger.info(f"Selected Best Model: {best_model_name}")

    logger.info(f"Confusion Matrix for {best_model_name}:\n{confusion_matrix(y_test, best_metrics['y_pred'])}")
    logger.info(f"Classification Report for {best_model_name}:\n{classification_report(y_test, best_metrics['y_pred'])}")

    model_save_path = "models/churn_model.pkl"
    joblib.dump(best_model, model_save_path)
    logger.info(f"Saved the best trained model to {model_save_path}")

    metadata = {
        'model_name': best_model_name,
        'model_class': best_model.__class__.__name__,
        'test_metrics': {
            'Accuracy': float(best_metrics['Test_Accuracy']),
            'Precision': float(best_metrics['Precision']),
            'Recall': float(best_metrics['Recall']),
            'F1_Score': float(best_metrics['F1_Score']),
            'ROC_AUC': float(best_metrics['ROC_AUC'])
        },
        'feature_names': list(X_train.columns)
    }
    joblib.dump(metadata, "models/model_metadata.joblib")

    logger.info("Computing SHAP explanations...")
    try:
        if "XGB" in best_model_name or "Random Forest" in best_model_name:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                shap_val_array = shap_values[1]
            else:
                shap_val_array = shap_values

            if len(shap_val_array.shape) == 3:
                shap_val_array = shap_val_array[:, :, 1]

            mean_abs_shap = np.mean(np.abs(shap_val_array), axis=0)
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': mean_abs_shap
            }).sort_values(by='importance', ascending=False)

            logger.info(f"\n=== SHAP Feature Importance (Top 10) ===\n{feature_importance.head(10).to_string(index=False)}")
            feature_importance.to_csv("data/processed/shap_importance.csv", index=False)

            joblib.dump(X_train.sample(100, random_state=42), "models/shap_background.joblib")

        else:
            logger.warning("SHAP computation not supported directly for non-tree models in this script pipeline.")

    except Exception as e:
        logger.error(f"Error computing SHAP values: {str(e)}")

if __name__ == "__main__":
    train_and_evaluate_all()
