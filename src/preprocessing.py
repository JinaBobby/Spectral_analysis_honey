from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_spectra(X, scaler=None, pca=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if pca is None:
        pca = PCA(n_components=0.95, random_state=42)
        X_final = pca.fit_transform(X_scaled)
    else:
        X_final = pca.transform(X_scaled)

    return X_final, scaler, pca