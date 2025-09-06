import streamlit as st
import joblib, json, numpy as np

# Load model and metadata
model = joblib.load('model.joblib')
with open('meta.json','r') as f:
    meta = json.load(f)

features = meta['features']
medians = meta['medians']

st.set_page_config(page_title="Water Potability Predictor", layout="centered")
st.title("Water Potability Predictor")
st.markdown("Enter physico-chemical parameters to predict whether water is potable.")

# Collect inputs dynamically
input_vals = []
cols = st.columns(2)
for i, feat in enumerate(features):
    default = medians.get(feat, 0)
    c = cols[i % 2]
    input_vals.append(c.number_input(f"{feat}", value=float(default)))

if st.button("Predict"):
    X = np.array([input_vals])
    pred = model.predict(X)[0]
    proba = None
    try:
        proba = model.predict_proba(X)[0][1]
    except:
        pass

    if pred == 1:
        st.success("Predicted: Potable ✅")
    else:
        st.error("Predicted: Not Potable ❌")

    if proba is not None:
        st.write(f"Probability potable: {proba:.2f}")
