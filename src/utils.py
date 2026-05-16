
from pathlib import Path
import joblib


def save_model(model, path="models/best_rf_model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path="models/best_rf_model.pkl"):
    return joblib.load(path)
