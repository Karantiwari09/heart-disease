# ❤️ Heart Disease Prediction App

A Streamlit app that uses an SVM model to predict the risk of heart disease based on medical input features.

## 🔍 Features
- Clean and responsive Streamlit UI
- Uses real UCI Heart Disease Dataset
- Predicts risk with SVM (Support Vector Machine)
- Shows probability of heart disease

## 📁 Files
- `heart_predict_app.py`: Main Streamlit app
- `model/svm_model.pkl`: Trained model
- `model/scaler.pkl`: Scaler for input features
- `requirements.txt`: Python dependencies

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run heart_predict_app.py
