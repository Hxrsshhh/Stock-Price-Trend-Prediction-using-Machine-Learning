import os
import pandas as pd
import matplotlib.pyplot as plt

from src.data import (
    download_stock_data,
    save_stock_data
)

from src.features import (
    create_features,
    BASELINE_FEATURES,
    INDICATOR_FEATURES,
    LAGGED_FEATURES
)

from src.models import (
    train_logistic_regression,
    train_random_forest,
    make_predictions
)

from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_all_metrics,
    plot_roc_curve
)


# ==========================================
# Configuration
# ==========================================

TICKER = "RELIANCE.NS"

START_DATE = "2015-01-01"

END_DATE = "2026-01-01"


# ==========================================
# Create required folders
# ==========================================

os.makedirs("data", exist_ok=True)

os.makedirs("plots", exist_ok=True)


# ==========================================
# STEP 1: Download data
# ==========================================

data = download_stock_data(
    ticker=TICKER,
    start=START_DATE,
    end=END_DATE
)

print("\nDownloaded data:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ==========================================
# STEP 2: Save raw data
# ==========================================

save_stock_data(
    data,
    "data/stock_data.csv"
)


# ==========================================
# STEP 3: Plot closing price
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    data["Date"],
    data["Close"]
)

plt.title(
    f"{TICKER} Historical Closing Price"
)

plt.xlabel("Date")

plt.ylabel("Price")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/stock_price.png",
    dpi=300
)

plt.show()


# ==========================================
# STEP 4: Create features
# ==========================================

data = create_features(data)

print("\nFeature dataset:")
print(data.head())

print("\nFeature columns:")
print(data.columns.tolist())


# ==========================================
# STEP 5: Create Train/Test Split
# ==========================================

split_index = int(
    len(data) * 0.8
)

y = data["Target"]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print("\nTarget distribution in test set:")

print(
    y_test.value_counts()
)

print("\nTarget percentages:")

print(
    y_test.value_counts(
        normalize=True
    ) * 100
)


# ==========================================
# HELPER FUNCTION
# ==========================================

def get_predictions_and_probabilities(
    model,
    X_test
):

    predictions = make_predictions(
        model,
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return predictions, probabilities


# ==========================================
# NAIVE BASELINE
# ==========================================

majority_class = y_train.mode()[0]

naive_predictions = [
    majority_class
] * len(y_test)

naive_results = evaluate_model(
    y_test,
    naive_predictions
)


print("\n")
print("=" * 50)
print("NAIVE BASELINE")
print("=" * 50)

for metric, value in naive_results.items():

    print(
        f"{metric}: {value:.4f}"
    )

print(
    "\nTraining samples:",
    len(y_train)
)

print(
    "Testing samples:",
    len(y_test)
)


# ==========================================
# EXPERIMENT 1
# BASELINE FEATURES
# ==========================================

print("\n")
print("=" * 50)
print("EXPERIMENT 1: BASELINE")
print("=" * 50)


X_baseline = data[
    BASELINE_FEATURES
]

X_baseline_train = X_baseline.iloc[
    :split_index
]

X_baseline_test = X_baseline.iloc[
    split_index:
]


# ------------------------------------------
# Baseline Logistic Regression
# ------------------------------------------

baseline_logistic = train_logistic_regression(
    X_baseline_train,
    y_train
)

(
    baseline_logistic_predictions,
    baseline_logistic_probabilities
) = get_predictions_and_probabilities(
    baseline_logistic,
    X_baseline_test
)

baseline_logistic_results = evaluate_model(
    y_test,
    baseline_logistic_predictions,
    baseline_logistic_probabilities
)


print(
    "\nBaseline - Logistic Regression"
)

for metric, value in baseline_logistic_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ------------------------------------------
# Baseline Random Forest
# ------------------------------------------

baseline_rf = train_random_forest(
    X_baseline_train,
    y_train
)

(
    baseline_rf_predictions,
    baseline_rf_probabilities
) = get_predictions_and_probabilities(
    baseline_rf,
    X_baseline_test
)

baseline_rf_results = evaluate_model(
    y_test,
    baseline_rf_predictions,
    baseline_rf_probabilities
)


print(
    "\nBaseline - Random Forest"
)

for metric, value in baseline_rf_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ==========================================
# EXPERIMENT 2
# HISTORICAL INDICATORS
# ==========================================

print("\n")
print("=" * 50)
print("EXPERIMENT 2: HISTORICAL INDICATORS")
print("=" * 50)


X_indicators = data[
    INDICATOR_FEATURES
]

X_indicators_train = X_indicators.iloc[
    :split_index
]

X_indicators_test = X_indicators.iloc[
    split_index:
]


# ------------------------------------------
# Indicator Logistic Regression
# ------------------------------------------

indicator_logistic = train_logistic_regression(
    X_indicators_train,
    y_train
)

(
    indicator_logistic_predictions,
    indicator_logistic_probabilities
) = get_predictions_and_probabilities(
    indicator_logistic,
    X_indicators_test
)

indicator_logistic_results = evaluate_model(
    y_test,
    indicator_logistic_predictions,
    indicator_logistic_probabilities
)


print(
    "\nIndicators - Logistic Regression"
)

for metric, value in indicator_logistic_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ------------------------------------------
# Indicator Random Forest
# ------------------------------------------

indicator_rf = train_random_forest(
    X_indicators_train,
    y_train
)

(
    indicator_rf_predictions,
    indicator_rf_probabilities
) = get_predictions_and_probabilities(
    indicator_rf,
    X_indicators_test
)

indicator_rf_results = evaluate_model(
    y_test,
    indicator_rf_predictions,
    indicator_rf_probabilities
)


print(
    "\nIndicators - Random Forest"
)

for metric, value in indicator_rf_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ==========================================
# EXPERIMENT 3
# INDICATORS + LAGGED FEATURES
# ==========================================

print("\n")
print("=" * 60)
print(
    "EXPERIMENT 3: INDICATORS + LAGGED FEATURES"
)
print("=" * 60)


X_lagged = data[
    LAGGED_FEATURES
]

X_lagged_train = X_lagged.iloc[
    :split_index
]

X_lagged_test = X_lagged.iloc[
    split_index:
]


# ------------------------------------------
# Lagged Logistic Regression
# ------------------------------------------

lagged_logistic = train_logistic_regression(
    X_lagged_train,
    y_train
)

(
    lagged_logistic_predictions,
    lagged_logistic_probabilities
) = get_predictions_and_probabilities(
    lagged_logistic,
    X_lagged_test
)

lagged_logistic_results = evaluate_model(
    y_test,
    lagged_logistic_predictions,
    lagged_logistic_probabilities
)


print(
    "\nIndicators + Lagged - Logistic Regression"
)

for metric, value in lagged_logistic_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ------------------------------------------
# Lagged Random Forest
# ------------------------------------------

lagged_rf = train_random_forest(
    X_lagged_train,
    y_train
)

(
    lagged_rf_predictions,
    lagged_rf_probabilities
) = get_predictions_and_probabilities(
    lagged_rf,
    X_lagged_test
)

lagged_rf_results = evaluate_model(
    y_test,
    lagged_rf_predictions,
    lagged_rf_probabilities
)


print(
    "\nIndicators + Lagged - Random Forest"
)

for metric, value in lagged_rf_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ==========================================
# FINAL MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(
    [
        naive_results,
        baseline_logistic_results,
        baseline_rf_results,
        indicator_logistic_results,
        indicator_rf_results,
        lagged_logistic_results,
        lagged_rf_results
    ],
    index=[
        "Naive Baseline",
        "Baseline - Logistic Regression",
        "Baseline - Random Forest",
        "Indicators - Logistic Regression",
        "Indicators - Random Forest",
        "Indicators + Lagged - Logistic Regression",
        "Indicators + Lagged - Random Forest"
    ]
)


print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.round(4)
)


# ==========================================
# Save comparison
# ==========================================

comparison.to_csv(
    "data/model_comparison.csv"
)


print(
    "\nResults saved to "
    "data/model_comparison.csv"
)


# ==========================================
# Confusion Matrix
# ==========================================

plot_confusion_matrix(
    y_test,
    lagged_rf_predictions,
    "Indicators + Lagged - Random Forest"
)


# ==========================================
# ROC Curve
# ==========================================

plot_roc_curve(
    y_test,
    lagged_rf_probabilities,
    "Indicators + Lagged - Random Forest"
)


# ==========================================
# CREATE COMPARISON GRAPHS
# ==========================================

plot_all_metrics(
    comparison
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print(
    "\nAll models have been trained and evaluated."
)

print(
    "Comparison saved to: "
    "data/model_comparison.csv"
)

print(
    "Plots saved to: plots/"
)