from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =================================================
# LOGISTIC REGRESSION
# =================================================

def train_logistic_regression(
    X_train,
    y_train
):
    """
    Train Logistic Regression.

    StandardScaler puts features on a similar
    numerical scale before training.
    """

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    return model


# =================================================
# RANDOM FOREST
# =================================================

def train_random_forest(
    X_train,
    y_train
):
    """
    Train Random Forest.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# =================================================
# MAKE PREDICTIONS
# =================================================

def make_predictions(
    model,
    X_test
):
    """
    Generate predictions from a trained model.
    """

    predictions = model.predict(
        X_test
    )

    return predictions