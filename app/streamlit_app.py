
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.model import get_default_model, tune_model
from src.utils import save_model, load_model

st.set_page_config(page_title="Hyperparameter Tuning App", layout="wide")

st.title("Day 26/180: Hyperparameter Tuning App 🚀")
st.write("Random Forest + GridSearchCV")

(X_train, X_test, y_train, y_test), data = load_data()

st.sidebar.header("What is Hyperparameter Tuning?")
st.sidebar.write("Hyperparameters are settings we choose before training.")
st.sidebar.write("GridSearchCV tries many combinations and picks the best one.")

if st.button("Train + Tune Model"):
    with st.spinner("Training default model..."):
        default_model = get_default_model()
        default_model.fit(X_train, y_train)
        default_pred = default_model.predict(X_test)
        default_acc = accuracy_score(y_test, default_pred)

    with st.spinner("Running GridSearchCV tuning..."):
        grid = tune_model(X_train, y_train)
        tuned_model = grid.best_estimator_
        tuned_pred = tuned_model.predict(X_test)
        tuned_acc = accuracy_score(y_test, tuned_pred)
        save_model(tuned_model)

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Accuracy", f"{default_acc:.2%}")
    col2.metric("Tuned Accuracy", f"{tuned_acc:.2%}")
    col3.metric("Best Model", "Random Forest")

    st.subheader("Best Hyperparameters")
    st.json(grid.best_params_)

    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(["Default Model", "Tuned Model"], [default_acc, tuned_acc])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, tuned_pred, target_names=data.target_names, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())

st.divider()
st.subheader("Sample Prediction")

try:
    model = load_model()
    st.success("Tuned model loaded successfully.")
except Exception:
    model = None
    st.warning("Train the model first by clicking 'Train + Tune Model'.")

sample = X_test.iloc[[0]]
st.write("Sample input:")
st.dataframe(sample)

if model is not None and st.button("Predict Sample"):
    pred = model.predict(sample)[0]
    st.success(f"Prediction: {data.target_names[pred]}")
