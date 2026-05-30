import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💰",
    layout="wide"
)

# Load Model
model = pickle.load(
    open("models/stacking_regressor.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

df = pd.read_csv("insurance (1).csv")

# Encoding
df["sex"] = df["sex"].map({
    "female": 0,
    "male": 1
})

df["smoker"] = df["smoker"].map({
    "no": 0,
    "yes": 1
})

df["region"] = df["region"].map({
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
})

st.title("💰 Insurance Charges Prediction")

st.markdown("---")

# Inputs

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        18,
        100,
        30
    )

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

with col2:

    children = st.number_input(
        "Children",
        0,
        10,
        1
    )

    smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

if st.button("Predict Charges"):

    sex_val = 1 if sex == "Male" else 0

    smoker_val = 1 if smoker == "Yes" else 0

    region_map = {
        "northeast": 0,
        "northwest": 1,
        "southeast": 2,
        "southwest": 3
    }

    input_data = np.array([[
        age,
        sex_val,
        bmi,
        children,
        smoker_val,
        region_map[region]
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(
        f"Predicted Insurance Charges: ${prediction:,.2f}"
    )

    chart_df = pd.DataFrame({
        "Charges": [prediction]
    })

    st.bar_chart(chart_df)

# Dataset Preview
st.markdown("---")

st.header("Dataset Preview")

st.dataframe(df.head())

# Charges Distribution
st.header("Charges Distribution")

fig, ax = plt.subplots()

ax.hist(df["charges"], bins=25)

ax.set_title("Insurance Charges Distribution")

st.pyplot(fig)

# BMI Distribution
st.header("BMI Distribution")

fig, ax = plt.subplots()

ax.hist(df["bmi"], bins=20)

ax.set_title("BMI Distribution")

st.pyplot(fig)

# Correlation Heatmap
st.header("Correlation Heatmap")

corr = df.corr()

fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(
    corr.columns,
    rotation=90
)

ax.set_yticklabels(
    corr.columns
)

fig.colorbar(im)

st.pyplot(fig)

# Model Information
st.info("""
Base Learners:
• Linear Regression
• Decision Tree Regressor
• Random Forest Regressor

Meta Learner:
• Ridge Regression

Algorithm:
• Stacking Regression
""")