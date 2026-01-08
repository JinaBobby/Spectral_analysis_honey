import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from src.preprocessing import preprocess_spectra

def train_svm(X, y):
    # Preprocess ONLY here
    X_processed, scaler, pca = preprocess_spectra(X)

    # Generate PCA plot
    y_numeric = [0 if label == 'Pure honey' else 1 for label in y]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_processed[:, 0], X_processed[:, 1], c=y_numeric, cmap='viridis', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Spectral Data')
    plt.colorbar(label='Class (0: Pure honey, 1: Adulterated honey)')
    plt.savefig('outputs/pca_plot.png')
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        C=10,
        gamma="scale"
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")

    return model, X_test, y_test