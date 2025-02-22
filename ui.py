import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load a model safely using joblib
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {model_path}. {str(e)}")
        return None

model_files = {
    "KNN": "knn_model.pkl",
    "SVM": "svm_model.pkl",
    "OCSVM": "ocsvm_model.pkl",
    "DBSCAN": "dbscan_model.pkl",
    "Hierarchial": "hierarchial_model.pkl",
    "Kmeans": "kmeans_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model_model.pkl",
    "Ensemble iForest + XGboost": "y_pred_xgb_model.pkl"
}

# Load models dynamically
models = {}
for model_name, file_name in model_files.items():
    model_path = os.path.join(current_dir, file_name)
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is not None:
            models[model_name] = model
    else:
        st.warning(f"⚠️ Model '{model_name}' not found! Skipping...")

st.title("🔬 Lead Conversion Prediction")
st.markdown("### **Select a model and enter your details to predict the likelihood of conversion**")

if len(models) == 0:
    st.error("No models available. Please retrain and save models with the individual-level features.")
    st.stop()

selected_model_name = st.selectbox("🛠 Choose a Prediction Model:", list(models.keys()))
selected_model = models[selected_model_name]

if not hasattr(selected_model, "predict"):
    st.error("❌ Error: The selected model is not valid. Please choose another model.")
    st.stop()

st.write("### **📊 Enter Your Personal & Lead Details**")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👵 Age (years)", min_value=18, max_value=100, step=1, help="Age of the lead.")
    is_senior = st.number_input("👵 Senior (Age > 65)", min_value=0, max_value=1, step=1, help="Is the lead a senior citizen?")
    credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=850, step=1, help="Credit score of the lead.")
    premium_amount = st.number_input("💰 Premium Amount (USD)", min_value=0.0, max_value=100000.0, step=100.0, help="Premium amount the lead is willing to pay.")
    time_to_conversion = st.number_input("⏳ Time to Conversion (days)", min_value=0, max_value=365, step=1, help="Time taken for lead conversion.")
    claims_frequency = st.number_input("🔄 Claims Frequency", min_value=0, max_value=10, step=1, help="Number of claims made by the lead.")
    claims_adjustment = st.number_input("⚖️ Claims Adjustment", min_value=0, max_value=1000, step=10, help="Adjustment to the claims made.")
    policy_adjustment = st.number_input("🔧 Policy Adjustment", min_value=0, max_value=1000, step=10, help="Adjustment to the policy.")
    total_discounts = st.number_input("💸 Total Discounts (%)", min_value=0, max_value=100, step=1, help="Total discounts applied to the premium.")
    time_since_first_contact = st.number_input("⏳ Time Since First Contact (days)", min_value=0, max_value=365, step=1, help="Days since the first contact.")
    website_visits = st.number_input("🌐 Website Visits", min_value=0, max_value=1000, step=1, help="Number of times the lead visited the website.")
    inquiries = st.number_input("❓ Inquiries", min_value=0, max_value=1000, step=1, help="Number of inquiries made by the lead.")
    quotes_requested = st.number_input("📑 Quotes Requested", min_value=0, max_value=1000, step=1, help="Number of quotes requested by the lead.")
    safe_driver_discount = st.number_input("🚗 Safe Driver Discount (%)", min_value=0, max_value=100, step=1, help="Discount for safe driving record.")
    multi_policy_discount = st.number_input("📦 Multi-Policy Discount (%)", min_value=0, max_value=100, step=1, help="Discount for having multiple policies.")
    
with col2:
    marital_status = st.selectbox("💍 Marital Status", options=["Single", "Married", "Divorced"], help="Marital status of the lead.")
    prior_insurance = st.selectbox("🛡️ Prior Insurance", options=["Yes", "No"], help="Does the lead have prior insurance?")
    policy_type = st.selectbox("📑 Policy Type", options=["Basic", "Standard", "Premium"], help="Type of policy the lead is interested in.")
    region = st.selectbox("🌍 Region", options=["North", "South", "East", "West"], help="Region of the lead.")
    source_of_lead = st.selectbox("📢 Source of Lead", options=["Online", "Referral", "Advertisement", "Direct"], help="Where did the lead come from?")
    marital_status_married = st.number_input("💍 Married Status", min_value=0, max_value=1, step=1, help="Is the lead married?")
    marital_status_single = st.number_input("💍 Single Status", min_value=0, max_value=1, step=1, help="Is the lead single?")
    marital_status_widowed = st.number_input("💍 Widowed Status", min_value=0, max_value=1, step=1, help="Is the lead widowed?")
    prior_insurance_1_year = st.number_input("🛡️ Prior Insurance < 1 year", min_value=0, max_value=1, step=1, help="Does the lead have prior insurance for less than 1 year?")
    prior_insurance_5_years = st.number_input("🛡️ Prior Insurance > 5 years", min_value=0, max_value=1, step=1, help="Does the lead have prior insurance for more than 5 years?")
    claims_severity_low = st.number_input("⚖️ Claims Severity Low", min_value=0, max_value=1, step=1, help="Is the claims severity low?")
    claims_severity_medium = st.number_input("⚖️ Claims Severity Medium", min_value=0, max_value=1, step=1, help="Is the claims severity medium?")
    policy_type_liability_only = st.number_input("📑 Policy Type Liability-Only", min_value=0, max_value=1, step=1, help="Is the policy type liability-only?")
    source_of_lead_online = st.number_input("📢 Source of Lead Online", min_value=0, max_value=1, step=1, help="Was the source of the lead online?")
    source_of_lead_referral = st.number_input("📢 Source of Lead Referral", min_value=0, max_value=1, step=1, help="Was the source of the lead referral?")
    region_suburban = st.number_input("🌍 Region Suburban", min_value=0, max_value=1, step=1, help="Is the lead from a suburban region?")
    region_urban = st.number_input("🌍 Region Urban", min_value=0, max_value=1, step=1, help="Is the lead from an urban region?")

# Feature preprocessing (one-hot encoding for categorical data)
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
prior_insurance_map = {"No": 0, "Yes": 1}
policy_type_map = {"Basic": 0, "Standard": 1, "Premium": 2}
region_map = {"North": 0, "South": 1, "East": 2, "West": 3}
source_of_lead_map = {"Online": 0, "Referral": 1, "Advertisement": 2, "Direct": 3}

# Map categorical values to numerical values
input_data = np.array([[
    age,
    is_senior,
    credit_score,
    premium_amount,
    time_to_conversion,
    claims_frequency,
    claims_adjustment,
    policy_adjustment,
    total_discounts,
    time_since_first_contact,
    website_visits,
    inquiries,
    quotes_requested,
    marital_status_map[marital_status],
    prior_insurance_map[prior_insurance],
    policy_type_map[policy_type],
    region_map[region],
    source_of_lead_map[source_of_lead],
    marital_status_married,
    marital_status_single,
    marital_status_widowed,
    prior_insurance_1_year,
    prior_insurance_5_years,
    claims_severity_low,
    claims_severity_medium,
    policy_type_liability_only,
    source_of_lead_online,
    source_of_lead_referral,
    region_suburban,
    region_urban,
    safe_driver_discount,
    multi_policy_discount   
]], dtype=np.float64)

# Check that the model expects the same number of features
expected_features = getattr(selected_model, "n_features_in_", input_data.shape[1])
if input_data.shape[1] != expected_features:
    st.error(
        f"❌ Feature Mismatch: The selected model expects {expected_features} features, "
        f"but got {input_data.shape[1]}. Please retrain the model with the individual-level features."
    )
    st.stop()

if st.button("🔍 Predict Conversion Likelihood"):
    try:
        prediction = selected_model.predict(input_data)[0]
        conversion_probability = max(0, min(100, prediction))
        st.success(f"🎯 Predicted Lead Conversion Likelihood: **{conversion_probability:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
