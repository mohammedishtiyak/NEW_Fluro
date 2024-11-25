import joblib

model_path = "best_regressor.joblib"
model = joblib.load(model_path)

if hasattr(model, "_sklearn_version"):
    print(f"Model was trained with scikit-learn version: {model._sklearn_version}")
else:
    print("The scikit-learn version used to train this model is not recorded.")
