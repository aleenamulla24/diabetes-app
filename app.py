import streamlit as st
import pickle
import pandas as pd

# Load model & features
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("Diabetes Progression Prediction")

# Inputs
age = st.slider("Age", 30, 50, 35)
time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
num_lab_procedures = st.slider("Lab Procedures", 1, 100, 40)
num_medications = st.slider("Medications", 1, 50, 10)
number_outpatient = st.slider("Outpatient Visits", 0, 20, 1)
number_emergency = st.slider("Emergency Visits", 0, 10, 0)
number_inpatient = st.slider("Inpatient Visits", 0, 10, 0)

diabetesMed = st.selectbox("Diabetes Medication", ["No", "Yes"])
# NEW INPUTS
A1Cresult = st.selectbox(
    "A1C Result (Blood Sugar Level)",
    ["None", "Normal", ">7", ">8"]
)

change = st.selectbox(
    "Change in Medication",
    ["No", "Yes"]
)
diabetesMed = 1 if diabetesMed == "Yes" else 0
A1C_map = {"None": 0, "Normal": 1, ">7": 2, ">8": 3}
change_map = {"No": 0, "Yes": 1}

A1C_encoded = A1C_map[A1Cresult]
change_encoded = change_map[change]

# Create input
input_data = pd.DataFrame([{
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "A1Cresult": A1C_encoded,
    "change": change_encoded,
    "diabetesMed": diabetesMed
}])
input_data = input_data[features]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("High Risk: Diabetes Progression")
    else:
        st.success("Low Risk")

    st.write(f"Probability: {probability:.2f}")