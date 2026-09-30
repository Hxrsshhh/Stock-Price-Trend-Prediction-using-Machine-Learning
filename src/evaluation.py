import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================
# Evaluate Model
# ============================================

def evaluate_model(y_true, y_pred, y_probability=None):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # ROC-AUC requires probability predictions
    if y_probability is not None:
        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )
    else:
        roc_auc = 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================
# Confusion Matrix
# ============================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name="Model"
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["DOWN", "UP"]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(ax=ax)

    ax.set_title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        f"plots/confusion_matrix_"
        f"{model_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


# ============================================
# ROC Curve
# ============================================

def plot_roc_curve(
    y_true,
    y_probability,
    model_name="Model"
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend()

    plt.tight_layout()

    filename = (
        f"plots/roc_curve_"
        f"{model_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


# ============================================
# Model Comparison Plot
# ============================================

def plot_model_comparison(
    comparison,
    metric="Accuracy"
):

    plt.figure(figsize=(11, 6))

    values = comparison[metric]

    bars = plt.bar(
        comparison.index,
        values
    )

    plt.title(
        f"Model Comparison - {metric}"
    )

    plt.ylabel(metric)
    plt.xlabel("Model")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    filename = (
        f"plots/model_comparison_"
        f"{metric.lower().replace(' ', '_')}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


# ============================================
# Plot All Metrics
# ============================================

def plot_all_metrics(comparison):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for metric in metrics:

        if metric in comparison.columns:

            plot_model_comparison(
                comparison,
                metric
            )