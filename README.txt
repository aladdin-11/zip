Cancer Prediction Web App
--------------------------
To run:

1. Place your .pkl model file and .csv dataset in this folder.
2. Install dependencies:
   pip install flask pandas joblib scikit-learn shap matplotlib

3. Run:
   python app.py

Then open: http://127.0.0.1:5000

This app includes:
- Login/Signup
- Model predictions with confidence
- SHAP explainability
- My Predictions history
- Escape Velocity styling
