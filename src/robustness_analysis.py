
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from src.feature_engineering import build_preprocessor


CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            C=1,
            max_iter=5000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    }


# 특정 변수를 제외할 수 있는 전처리 함수
def get_preprocessor(
    imputation="median",
    excluded_features=None
):

    if excluded_features is None:
        excluded_features = []

    preprocessor = build_preprocessor(
        imputation=imputation,
        encoding="onehot",
        scaling="standard"
    )

    # 제외할 변수를 전처리 대상에서 제거
    preprocessor.transformers = [
        (
            name,
            transformer,
            [
                col for col in columns
                if col not in excluded_features
            ]
        )
        for name, transformer, columns
        in preprocessor.transformers
    ]

    return preprocessor


# 1. 결측치 강건성 분석
def run_missing_robustness(X_train, y_train):

    print("\n=== Missing Data Robustness ===")

    missing_rates = [
        0.0, 0.1, 0.3, 0.5, 0.7
    ]

    strategies = [
        "mean",
        "median",
        "most_frequent"
    ]

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        CV.split(X_train, y_train)
    ):

        X_fit = X_train.iloc[train_idx].copy()
        y_fit = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx].copy()
        y_val = y_train.iloc[val_idx]

        # 기존에 나이가 존재하는 데이터만 선택
        observed = X_val.index[
            X_val["Age"].notna()
        ].to_numpy()

        rng = np.random.default_rng(42 + fold)

        shuffled = rng.permutation(observed)

        for strategy in strategies:

            for model_name, model in get_models().items():

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        get_preprocessor(
                            imputation=strategy,
                            excluded_features=["AgeGroup"]
                        )
                    ),
                    ("model", model)
                ])

                # 원본 학습 데이터로 모델 학습
                pipeline.fit(X_fit, y_fit)

                for rate in missing_rates:

                    X_corrupted = X_val.copy()

                    count = int(
                        len(shuffled) * rate
                    )

                    missing_indices = shuffled[:count]

                    # 검증 데이터에 추가 결측치 생성
                    X_corrupted.loc[
                        missing_indices, "Age"
                    ] = np.nan

                    y_pred = pipeline.predict(
                        X_corrupted
                    )

                    y_prob = pipeline.predict_proba(
                        X_corrupted
                    )[:, 1]

                    results.append({
                        "Fold": fold + 1,
                        "Model": model_name,
                        "Imputation": strategy,

                        "Additional Missing (%)":
                            rate * 100,

                        "Actual Missing (%)":
                            X_corrupted["Age"]
                            .isna().mean() * 100,

                        "Accuracy": accuracy_score(
                            y_val, y_pred
                        ),

                        "F1-score": f1_score(
                            y_val,
                            y_pred,
                            zero_division=0
                        ),

                        "ROC-AUC": roc_auc_score(
                            y_val, y_prob
                        )
                    })

        print(f"Fold {fold + 1} completed.")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/missing_robustness_raw.csv",
        index=False
    )

    # 교차 검증 평균 및 표준편차 계산
    summary = results_df.groupby([
        "Model",
        "Imputation",
        "Additional Missing (%)"
    ]).agg(
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_F1=("F1-score", "mean"),
        Std_F1=("F1-score", "std"),
        Mean_ROC_AUC=("ROC-AUC", "mean"),
        Actual_Missing=("Actual Missing (%)", "mean")
    ).reset_index()

    # 추가 결측치 0% 대비 F1-score 변화
    baseline = summary[
        summary["Additional Missing (%)"] == 0
    ][
        ["Model", "Imputation", "Mean_F1"]
    ].rename(
        columns={"Mean_F1": "Baseline_F1"}
    )

    summary = summary.merge(
        baseline,
        on=["Model", "Imputation"]
    )

    summary["F1_Drop"] = (
        summary["Baseline_F1"]
        - summary["Mean_F1"]
    )

    summary.to_csv(
        "results/missing_robustness_summary.csv",
        index=False
    )

    print("\n=== Missing Robustness Results ===")
    print(summary.round(4).to_string(index=False))

    # 모델별 결측치 강건성 그래프
    for model_name in get_models():

        subset = summary[
            summary["Model"] == model_name
        ]

        plt.figure(figsize=(10, 6))

        for strategy in strategies:

            data = subset[
                subset["Imputation"] == strategy
            ]

            plt.errorbar(
                data["Additional Missing (%)"],
                data["Mean_F1"],
                yerr=data["Std_F1"],
                marker="o",
                capsize=4,
                label=strategy
            )

        plt.title(
            f"Missing Data Robustness - {model_name}"
        )

        plt.xlabel("Additional Missing Rate (%)")
        plt.ylabel("Mean CV F1-score")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            f"results/figures/robustness_"
            f"{model_name.replace(' ', '_')}.png",
            dpi=300
        )

        plt.close()

    return summary


# 2. 파생 변수별 Ablation Study
def run_ablation_study(X_train, y_train):

    print("\n=== Ablation Study ===")

    derived_features = [
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "AgeGroup",
        "Title",
        "HasCabin"
    ]

    experiments = {
        "All Features": []
    }

    for feature in derived_features:

        experiments[f"Without {feature}"] = [
            feature
        ]

    results = []

    for experiment, excluded in experiments.items():

        for model_name, model in get_models().items():

            pipeline = Pipeline([
                (
                    "preprocessor",
                    get_preprocessor(
                        excluded_features=excluded
                    )
                ),
                ("model", model)
            ])

            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=CV,
                scoring=[
                    "accuracy",
                    "f1",
                    "roc_auc"
                ],
                n_jobs=1,
                return_train_score=True
            )

            results.append({
                "Experiment": experiment,
                "Model": model_name,

                "CV Accuracy":
                    scores["test_accuracy"].mean(),

                "CV F1":
                    scores["test_f1"].mean(),

                "CV F1 Std":
                    scores["test_f1"].std(),

                "CV ROC-AUC":
                    scores["test_roc_auc"].mean(),

                "Train Accuracy":
                    scores["train_accuracy"].mean()
            })

            print(
                f"{experiment} | "
                f"{model_name} | "
                f"F1: "
                f"{scores['test_f1'].mean():.4f}"
            )

    results_df = pd.DataFrame(results)

    baseline = results_df[
        results_df["Experiment"] == "All Features"
    ][
        ["Model", "CV F1"]
    ].rename(
        columns={"CV F1": "Baseline F1"}
    )

    results_df = results_df.merge(
        baseline,
        on="Model"
    )

    # 양수이면 변수를 제거했을 때 성능 감소
    results_df["Feature Contribution"] = (
        results_df["Baseline F1"]
        - results_df["CV F1"]
    )

    results_df.to_csv(
        "results/ablation_results.csv",
        index=False
    )

    print("\n=== Ablation Results ===")

    print(
        results_df.round(4).to_string(index=False)
    )

    plot_data = results_df[
        results_df["Experiment"] != "All Features"
    ]

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=plot_data,
        x="Feature Contribution",
        y="Experiment",
        hue="Model"
    )

    plt.axvline(
        0,
        color="black",
        linewidth=1
    )

    plt.title("Derived Feature Contribution")
    plt.xlabel("Baseline F1 - Ablation F1")
    plt.tight_layout()

    plt.savefig(
        "results/figures/ablation_comparison.png",
        dpi=300
    )

    plt.close()

    return results_df
