
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('xgboost_model_VIN.pkl')


# Define feature names
feature_names = [
    "Follow-up HPV", "Follow-up Cytology", "Co-VaIN", "TZ III", "Age"
]

# Streamlit user interface
st.title("VaIN Prediction System")

# age: numerical input
#age = st.number_input("Age:", min_value=1, max_value=120, value=50)

age = st.selectbox("Age (0 =< 48, 1 => 48):", options=[0, 1], format_func=lambda x: '<48 (0)' if x == 0 else '>48 (1)')


HPV = st.selectbox("Follow-up HPV (0 = Negative, 1 = Positive):", options=[0, 1], format_func=lambda x: '<48 (0)' if x == 0 else '>48 (1)')

# sex: categorical selection
TCT = st.selectbox("Follow-up Cytology (0 = Negative, 1 = Positive):", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

# cp: categorical selection
VaIN = st.selectbox("Co-VaIN:", options=[0, 1], format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

# fbs: categorical selection
TZ = st.selectbox("Transformation Zone Type(0= Type I and II, 1 = Type III):", options=[0, 1], format_func=lambda x: 'Type I and II (0)' if x == 0 else 'Type III (1)')



# Process inputs and make predictions
feature_values = [HPV, TCT, VaIN, TZ, age]
features = np.array([feature_values])


if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our predictive model, you have a high risk of developing VaIN after conization, with an estimated probability of {probability:.1f}%.  "
            f"Although this result is an estimate based on the model's calculations, it suggests a significant potential risk. "
            "I strongly recommend that you consult a gynecological specialist as soon as possible for further evaluation, accurate diagnosis, "
            "and timely management or treatment if necessary. "

        )
    else:
        advice = (
            f"According to our predictive model, your risk of developing VaIN after conization is relatively low, with an estimated probability of {probability:.1f}%. "
            f"However, it remains very important to maintain a healthy lifestyle and undergo regular health screenings. "
            "We recommend scheduling periodic check-ups and promptly consulting a doctor if you experience any concerning symptoms. "

        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)

    st.image("shap_force_plot.png")
