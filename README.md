
# Day 25/180 — Hyperparameter Tuning Project

Mini project to explain Hyperparameter Tuning using:

- Random Forest Classifier
- GridSearchCV
- Breast Cancer dataset from scikit-learn
- Streamlit dashboard

## Run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

## Project Structure

```text
hyperparameter_tuning_rf_project/
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```
