import joblib
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_spectra

def predict_new_sample(sample_path):
    model = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    df = pd.read_excel(sample_path, engine="openpyxl")

    print("DEBUG new sample shape:", df.shape)

    X_new = df.iloc[:, 1:].values   # spectral columns only

    X_new_processed, _, _ = preprocess_spectra(
        X_new,
        scaler=scaler,
        pca=pca
    )

    prediction = model.predict(X_new_processed)[0]
    prob = model.predict_proba(X_new_processed)[0]
    confidence = np.max(prob) * 100

    print("\n Prediction Result")
    print("-------------------")
    print(f"Predicted class : {prediction}")
    print(f"Confidence      : {confidence:.2f}%")

    return prediction, confidence