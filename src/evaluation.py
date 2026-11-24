import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),

        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),

        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        ),

        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }


def save_results(results):

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_dir / "experiment_results.csv",
        index=False
    )

    print("\n=== Experiment Results ===")
    print(results_df.round(4).to_string(index=False))

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    for metric in metrics:

        plt.figure(figsize=(15, 6))

        sns.barplot(
            data=results_df,
            x="Experiment",
            y=metric,
            hue="Model"
        )

        plt.title(f"{metric} Comparison")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            figures_dir / f"comparison_{metric}.png",
            dpi=300
        )

        plt.close()

    print("\nEvaluation results saved.")

    return results_df