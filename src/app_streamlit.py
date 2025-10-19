# src/app_streamlit.py
import streamlit as st
from predict import predict_baseline, predict_lstm

st.set_page_config(page_title="DeepCSAT – E-commerce CSAT Predictor", page_icon="🛒")

st.title("🧠 DeepCSAT – Customer Satisfaction Prediction")
st.write("Predict whether a customer is **Satisfied 😄** or **Not Satisfied 😡** based on their feedback message.")

# Input box
text = st.text_area("🗣 Enter Customer Remarks", height=200, placeholder="Type or paste a customer message here...")

# Model selection
model_choice = st.selectbox("Select Model", ["Baseline (TF-IDF + Logistic Regression)", "Deep (Bi-LSTM)"])

if st.button("🚀 Predict Satisfaction"):
    if not text.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing sentiment..."):
            if "Baseline" in model_choice:
                pred = predict_baseline([text])[0]
            else:
                pred = predict_lstm([text])[0]

        # Display friendly message
        if int(pred) == 1:
            st.success("✅ Customer is **Satisfied 😄**")
        else:
            st.error("❌ Customer is **Not Satisfied 😡**")

st.markdown("---")
st.caption("Developed with ❤️ using Python, Streamlit, and TensorFlow")
