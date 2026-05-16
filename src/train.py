
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.model import get_default_model, tune_model
from src.utils import save_model


def main():
    (X_train, X_test, y_train, y_test), data = load_data()

    print("Training default Random Forest...")
    default_model = get_default_model()
    default_model.fit(X_train, y_train)
    default_pred = default_model.predict(X_test)
    default_acc = accuracy_score(y_test, default_pred)

    print("Running GridSearchCV...")
    grid = tune_model(X_train, y_train)

    tuned_model = grid.best_estimator_
    tuned_pred = tuned_model.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_pred)

    save_model(tuned_model)

    print("\nDefault Accuracy:", round(default_acc, 4))
    print("Tuned Accuracy:", round(tuned_acc, 4))
    print("Best Params:", grid.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, tuned_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()
