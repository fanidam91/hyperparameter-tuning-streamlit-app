
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_default_model():
    return RandomForestClassifier(random_state=42)


def get_param_grid():
    return {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }


def tune_model(X_train, y_train):
    grid = GridSearchCV(
        estimator=get_default_model(),
        param_grid=get_param_grid(),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid
