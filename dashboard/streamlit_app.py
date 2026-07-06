import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0d0f14;
        color: #f3f4f6;
    }

    /* Title and Headers */
    .title-text {
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .subtitle-text {
        font-weight: 400;
        color: #9ca3af;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Cards & Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.3);
    }

    /* Metrics block */
    .kpi-container {
        display: flex;
        justify-content: space-around;
        gap: 15px;
        margin-bottom: 25px;
    }

    .kpi-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk Levels styling */
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #fca5a5;
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #fcd34d;
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #6ee7b7;
    }

    /* Form inputs and buttons styling */
    div[data-baseweb="select"] > div, input, div[data-baseweb="input"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f3f4f6 !important;
        border-radius: 8px !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }

    /* Recommendation Alert Box */
    .rec-box {
        background-color: rgba(59, 130, 246, 0.07);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        color: #e0f2fe;
    }
</style>
""", unsafe_allow_html=True)

def query_api_predict(customer_data: dict) -> dict:
    api_url = os.environ.get("API_URL", "http://localhost:8000/predict")
    try:
        response = requests.post(api_url, json=customer_data, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: Received status code {response.status_code}. Detail: {response.text}")
            return None
    except requests.exceptions.RequestException:
        try:
            from src.predict import ChurnPredictor
            predictor = ChurnPredictor()
            return predictor.predict(customer_data)
        except Exception as local_ex:
            st.error(f"Fallback to local prediction engine failed: {str(local_ex)}")
            st.warning("Make sure the backend API is running (`uvicorn api.app:app`) or models are trained (`python src/train.py`).")
            return None

@st.cache_data
def load_dataset_summary():
    raw_path = "data/raw/Telco_customer_churn.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        df['Total Charges'] = pd.to_numeric(df['Total Charges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
        return df
    return None

df_raw = load_dataset_summary()

metadata = None
if os.path.exists("models/model_metadata.joblib"):
    try:
        metadata = joblib.load("models/model_metadata.joblib")
    except Exception:
        pass

st.markdown("<div class='title-text'>Customer Churn Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Analyze subscriber behaviors, identify churn risks in real-time, and get smart recommendations for customer retention.</div>", unsafe_allow_html=True)

tab_predict, tab_analytics = st.tabs(["🔮 Real-Time Customer Predictor", "📊 Business Overview & Metrics"])

with tab_predict:
    st.markdown("<div class='glass-card'><h3>Enter Customer Details</h3>", unsafe_allow_html=True)

    st.subheader("Demographics")
    col1_1, col1_2, col1_3 = st.columns(3)
    with col1_1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col1_2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col1_3:
        partner = st.selectbox("Partner", ["No", "Yes"])

    col2_1, col2_2, col2_3 = st.columns(3)
    with col2_1:
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    with col2_2:
        cltv = st.number_input("Customer Lifetime Value (CLTV)", min_value=1000, max_value=7000, value=3500, step=10)
    with col2_3:
        st.write("")

    st.subheader("Subscription & Service")
    col3_1, col3_2, col3_3 = st.columns(3)
    with col3_1:
        tenure = st.number_input("Tenure Months", min_value=0, max_value=72, value=12)
    with col3_2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with col3_3:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    col4_1, col4_2, col4_3 = st.columns(3)
    with col4_1:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with col4_2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with col4_3:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    col5_1, col5_2, col5_3 = st.columns(3)
    with col5_1:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with col5_2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with col5_3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    col6_1, col6_2, col6_3 = st.columns(3)
    with col6_1:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    with col6_2:
        st.write("")
    with col6_3:
        st.write("")

    st.subheader("Billing & Contract")
    col7_1, col7_2, col7_3 = st.columns(3)
    with col7_1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with col7_2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col7_3:
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    col8_1, col8_2, col8_3 = st.columns(3)
    with col8_1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=75.0, step=0.5)
    with col8_2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=900.0, step=10.0)
    with col8_3:
        st.write("")

    st.markdown("</div>", unsafe_allow_html=True)

    cust_data = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": int(tenure),
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": float(monthly_charges),
        "Total Charges": float(total_charges),
        "CLTV": int(cltv)
    }

    if st.button("🚀 Analyze Churn Risk"):
        with st.spinner("Processing customer details and running AI model..."):
            result = query_api_predict(cust_data)

        if result:
            pred = result["prediction"]
            prob = result["probability"]
            risk_score = result["risk_score"]
            risk_level = result["risk_level"]
            recs = result["recommendations"]
            shap_exp = result.get("shap_explanation", {})

            st.markdown("---")
            st.subheader("🎯 Prediction & Analysis Report")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                risk_style = "risk-low"
                risk_color = "#10b981"
                if risk_level == "Medium":
                    risk_style = "risk-medium"
                    risk_color = "#f59e0b"
                elif risk_level == "High":
                    risk_style = "risk-high"
                    risk_color = "#ef4444"

                st.markdown(f"""
                <div class='{risk_style}'>
                    <h3 style='margin-top:0; color:inherit;'>Churn Status: {pred.upper()}</h3>
                    <div style='font-size: 3.5rem; font-weight:800;'>{risk_score}%</div>
                    <div style='font-size: 1.1rem; font-weight:600; text-transform:uppercase;'>Risk Level: {risk_level}</div>
                    <p style='margin-bottom:0; font-size:0.95rem; opacity:0.9;'>
                        This customer has a {risk_score}% probability of cancelling their services based on historical behaviors.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Customer Risk Score Index (0-100)"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#f3f4f6"},
                        'bar': {'color': risk_color},
                        'bgcolor': "rgba(255, 255, 255, 0.05)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255, 255, 255, 0.1)",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                            {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#f3f4f6", 'family': "Outfit"},
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.markdown("<div class='glass-card'><h4>💡 Actionable Retention Recommendations</h4>", unsafe_allow_html=True)
                for rec in recs:
                    st.markdown(f"<div class='rec-box'>{rec}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if shap_exp and (shap_exp.get("risk_drivers") or shap_exp.get("safety_factors")):
                st.markdown("<div class='glass-card'><h4>🔍 What is Driving this Risk Score? (SHAP Explainer)</h4>", unsafe_allow_html=True)

                drivers = shap_exp.get("risk_drivers", [])
                safeties = shap_exp.get("safety_factors", [])

                shap_df_list = []
                for item in drivers:
                    shap_df_list.append({"Feature": item["feature"], "Impact": item["value"], "Type": "Risk Increaser (Bad)"})
                for item in safeties:
                    shap_df_list.append({"Feature": item["feature"], "Impact": item["value"], "Type": "Retention Factor (Good)"})

                if shap_df_list:
                    df_shap = pd.DataFrame(shap_df_list)
                    fig_shap = px.bar(
                        df_shap,
                        y="Feature",
                        x="Impact",
                        color="Type",
                        orientation="h",
                        color_discrete_map={"Risk Increaser (Bad)": "#ef4444", "Retention Factor (Good)": "#10b981"},
                        title="Key Factor Impact Analysis (SHAP Values)",
                        labels={"Impact": "Contribution Magnitude"}
                    )
                    fig_shap.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#f3f4f6", 'family': "Outfit"},
                        yaxis={'categoryorder': 'total ascending'},
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

with tab_analytics:
    if df_raw is not None:
        total_cust = len(df_raw)
        churn_rate = (df_raw['Churn Value'] == 1).mean() * 100
        avg_monthly = df_raw['Monthly Charges'].mean()
        avg_tenure = df_raw['Tenure Months'].mean()

        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-card'>
                <div class='kpi-value' style='color:#3b82f6;'>{total_cust:,}</div>
                <div class='kpi-label'>Total Customers</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value' style='color:#ef4444;'>{churn_rate:.1f}%</div>
                <div class='kpi-label'>Churn Rate</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value' style='color:#10b981;'>${avg_monthly:.2f}</div>
                <div class='kpi-label'>Avg Monthly Bill</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value' style='color:#a855f7;'>{avg_tenure:.1f} mo</div>
                <div class='kpi-label'>Avg Tenure</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            contract_churn = df_raw.groupby('Contract')['Churn Value'].mean().reset_index()
            contract_churn['Churn Rate (%)'] = contract_churn['Churn Value'] * 100

            fig1 = px.bar(
                contract_churn,
                x='Contract',
                y='Churn Rate (%)',
                text='Churn Rate (%)',
                title="Churn Rate by Contract Type",
                color_discrete_sequence=['#3b82f6']
            )
            fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                height=300
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            internet_churn = df_raw.groupby('Internet Service')['Churn Value'].mean().reset_index()
            internet_churn['Churn Rate (%)'] = internet_churn['Churn Value'] * 100

            fig2 = px.bar(
                internet_churn,
                x='Internet Service',
                y='Churn Rate (%)',
                text='Churn Rate (%)',
                title="Churn Rate by Internet Service Provider",
                color_discrete_sequence=['#10b981']
            )
            fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            fig3 = px.scatter(
                df_raw.sample(min(1000, len(df_raw)), random_state=42),
                x='Tenure Months',
                y='Monthly Charges',
                color='Churn Label',
                color_discrete_map={'Yes': '#ef4444', 'No': '#3b82f6'},
                title="Tenure vs Monthly Charges (Colored by Churn)",
                opacity=0.6
            )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                height=300
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if os.path.exists("data/processed/shap_importance.csv"):
                df_imp = pd.read_csv("data/processed/shap_importance.csv").head(8)
                fig_imp = px.bar(
                    df_imp,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Global Model Feature Importance (SHAP values)',
                    color_discrete_sequence=['#a855f7']
                )
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#f3f4f6", 'family': "Outfit"},
                    yaxis={'categoryorder':'total ascending'},
                    height=300
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #6b7280;'>Train the model to view feature importances.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Telco Customer Churn dataset not found. Please place `Telco_customer_churn.csv` in `data/raw/` to view analytics.")

st.sidebar.markdown("### 🔮 System Control")
if metadata:
    st.sidebar.success(f"Model Status: LOADED ({metadata.get('model_name', 'Tuned Classifier')})")
    metrics = metadata.get("test_metrics", {})
    st.sidebar.markdown(f"""
    **Model Performance:**
    - Accuracy: `{metrics.get('Accuracy', 0):.2%}`
    - Precision: `{metrics.get('Precision', 0):.2%}`
    - Recall (Churn): `{metrics.get('Recall', 0):.2%}`
    - F1 Score: `{metrics.get('F1_Score', 0):.2%}`
    - ROC AUC: `{metrics.get('ROC_AUC', 0):.2%}`
    """)
else:
    st.sidebar.warning("Model Status: NOT TRAINED YET")
    st.sidebar.info("Run training in notebooks or via terminal: `python src/train.py` to activate predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 API Connection")
st.sidebar.code("POST /predict\nHost: localhost:8000")
st.sidebar.markdown("""
This application runs an end-to-end churn classification pipeline.
""")
