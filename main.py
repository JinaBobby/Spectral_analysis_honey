from src.load_data import load_spectral_data
from src.train_model import train_svm
from src.evaluate_model import evaluate
from src.predict_sample import predict_new_sample

path = "data/data.xlsx"
X, y = load_spectral_data(path)

print("Loaded data shape:", X.shape)
print(y.value_counts())

model, X_test, y_test = train_svm(X, y)
evaluate(model, X_test, y_test)

predict_new_sample("data/new_sample_a.xlsx")