# DeepCSAT — Ecommerce
Customer Satisfaction Prediction for ecommerce customer support messages.

## Setup
1. Create virtual env: `python -m venv venv && source venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Put dataset in `data/eCommerce_Customer_support_data.csv`

## Run
- Train baseline: `python src/train_baseline.py`
- Train LSTM: `python src/train_lstm.py`
- Predict: `python src/predict.py --text "..." --model baseline`
- Streamlit app: `streamlit run src/app_streamlit.py`
