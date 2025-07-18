import streamlit as st
import pandas as pd
import pickle

st.title("SME Sales Prediction App")

@st.cache(allow_output_mutation=True)
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

uploaded_file = st.file_uploader("Upload your SME Sales CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview", data.head())

    try:
        predictions = model.predict(data)
        data["Predicted"] = predictions
        st.subheader("Predictions")
        st.write(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")