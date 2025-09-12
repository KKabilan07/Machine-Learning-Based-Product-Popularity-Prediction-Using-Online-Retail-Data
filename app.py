import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Caching resources
@st.cache_resource
def load_models_and_scaler():
    classification_model = load('models/classification.pkl')
    regression_model = load('models/regression.pkl')
    scaler = load('models/scaler_classification.joblib')
    return classification_model, regression_model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('aggregated_products.csv')

classification_model, regression_model, scaler = load_models_and_scaler()
data = load_data()

st.set_page_config(page_title="Retail Prediction App", layout="wide")

# Sidebar Navigation
st.sidebar.title("🛠️ Menu")
page = st.sidebar.radio("Navigate", ["🏠 Home", "⭐ Popularity Prediction", "💰 Sales Forecast", "ℹ️ About"])

# 🏠 Home Page
if page == "🏠 Home":
    st.title("📊 Welcome to Product Popularity & Sales Prediction App")
    st.write("""
    Use the sidebar to navigate between:
    - Popularity Prediction  
    - Future Sales Forecast  
    - About section  
    """)

# ⭐ Popularity Prediction Page
elif page == "⭐ Popularity Prediction":
    st.title("⭐ Product Popularity Prediction")

    st.markdown("Enter the key product features below (all in one compact view):")

    with st.form('classification_form', clear_on_submit=False):
        cols = st.columns(4)

        total_quantity = cols[0].number_input('Total Quantity Sold', min_value=0, step=10)
        num_transactions = cols[1].number_input('Number of Transactions', min_value=0, step=5)
        num_customers = cols[2].number_input('Number of Unique Customers', min_value=0, step=5)
        revenue = cols[3].number_input('Total Revenue', min_value=0.0, step=0.1)

        submit_classification = st.form_submit_button('🔍 Predict Popularity')

        if submit_classification:
            input_features = np.array([[total_quantity, num_transactions, num_customers, revenue]])
            input_scaled = scaler.transform(input_features)

            prediction = classification_model.predict(input_scaled)[0]
            proba = classification_model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.success(f"⭐ Product is **POPULAR** (Probability: {proba:.2f})")
            else:
                st.warning(f"❌ Product is **NOT POPULAR** (Probability: {proba:.2f})")

# 💰 Future Sales Forecast Page
elif page == "💰 Sales Forecast":
    st.title("💰 Future Sales Forecast")

    st.markdown("Enter historical sales data in a compact form:")

    with st.form('regression_form', clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        lag1 = col1.number_input('Lag1 (Previous Month Sales)', min_value=0.0, step=1.0)
        lag2 = col1.number_input('Lag2 (Two Months Ago Sales)', min_value=0.0, step=1.0)

        revenue = col2.number_input('Current Month Revenue', min_value=0.0, step=0.01)
        invoice_count = col2.number_input('Number of Unique Invoices', min_value=0, step=1)

        avg_price = col3.number_input('Average Price of Product', min_value=0.0, step=0.01)

        submit_regression = st.form_submit_button('📈 Predict Future Sales')

        if submit_regression:
            input_features = np.array([[lag1, lag2, revenue, invoice_count, avg_price]])
            predicted_sales = regression_model.predict(input_features)[0]

            st.success(f"📊 Predicted Future Sales: **{predicted_sales:.2f} units**")

# ℹ️ About Page
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.write("""
    ⚡ Predict product popularity & forecast future sales using ML models.  
    ✅ Clean UI design: Sidebar + Compact Input Forms  
    🚀 Developed by Team C7 AIE 
    """)
