import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -----------------------------
# Load trained models
# -----------------------------

xgb_model = joblib.load("xgb_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
seg_scaler = joblib.load("seg_scaler.pkl")
model_features = joblib.load("model_features.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Credit Card Risk Analyzer",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Risk Analyzer")
st.write("Predict credit default risk and identify customer segments.")


# -----------------------------
# Customer inputs
# -----------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input(
        "Credit Limit",
        min_value=0.0,
        value=50000.0
    )

    sex = st.selectbox(
        "Sex",
        [1, 2]
    )

    education = st.selectbox(
        "Education",
        [1, 2, 3, 4]
    )

with col2:
    marriage = st.selectbox(
        "Marriage",
        [1, 2, 3]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col3:
    pay_0 = st.number_input("PAY_0", value=0, step=1)
    pay_2 = st.number_input("PAY_2", value=0, step=1)
    pay_3 = st.number_input("PAY_3", value=0, step=1)


st.subheader("Payment History")

col1, col2, col3 = st.columns(3)

with col1:
    pay_4 = st.number_input("PAY_4", value=0, step=1)
    pay_5 = st.number_input("PAY_5", value=0, step=1)

with col2:
    pay_6 = st.number_input("PAY_6", value=0, step=1)

with col3:
    st.info(
        "Payment status values follow the original dataset coding."
    )


st.subheader("Bill Amounts")

col1, col2, col3 = st.columns(3)

with col1:
    bill_amt1 = st.number_input("BILL_AMT1", value=0.0)
    bill_amt2 = st.number_input("BILL_AMT2", value=0.0)

with col2:
    bill_amt3 = st.number_input("BILL_AMT3", value=0.0)
    bill_amt4 = st.number_input("BILL_AMT4", value=0.0)

with col3:
    bill_amt5 = st.number_input("BILL_AMT5", value=0.0)
    bill_amt6 = st.number_input("BILL_AMT6", value=0.0)


st.subheader("Payment Amounts")

col1, col2, col3 = st.columns(3)

with col1:
    pay_amt1 = st.number_input("PAY_AMT1", value=0.0)
    pay_amt2 = st.number_input("PAY_AMT2", value=0.0)

with col2:
    pay_amt3 = st.number_input("PAY_AMT3", value=0.0)
    pay_amt4 = st.number_input("PAY_AMT4", value=0.0)

with col3:
    pay_amt5 = st.number_input("PAY_AMT5", value=0.0)
    pay_amt6 = st.number_input("PAY_AMT6", value=0.0)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Risk", type="primary"):

    input_data = pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6
    }])

    # Keep exact training feature order
    input_data = input_data[model_features]

    # Default probability
    probability = xgb_model.predict_proba(input_data)[0, 1]

    prediction = xgb_model.predict(input_data)[0]

    st.divider()

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

    with col2:
        if prediction == 1:
            st.error("⚠️ Higher Default Risk")
        else:
            st.success("✅ Lower Default Risk")


    # -----------------------------
    # Customer Segmentation
    # -----------------------------

    st.header("Customer Segment")

    segment_data = input_data[
        [
            "LIMIT_BAL",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3"
        ]
    ]

    segment_scaled = seg_scaler.transform(segment_data)

    cluster = kmeans_model.predict(segment_scaled)[0]

    st.info(f"Customer belongs to Segment {cluster}")