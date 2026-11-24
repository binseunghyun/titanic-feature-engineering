
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)

from src.feature_engineering import build_preprocessor
from src.feature_selection import build_feature_selector
from src.evaluation import evaluate_model

# 교차 검증 설정
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 공통 전처리 방식 (Exp-4 기준)
def get_preprocessor():

    return build_preprocessor(
        imputation="median",
        encoding="onehot",
        scaling="standard"
    )

# 1. GridSearchCV 및 과적합 분석
def run_grid_search(X_train, X_test, y_train, y_test):

    models = {

        "Logistic Regression": (
            LogisticRegression(
                max_iter=5000,
                random_state=42
            ),

            {
                "model__C": [
                    0.01, 0.1, 1, 10, 100
                ],
                "model__class_weight": [
                    None, "balanced"
                ]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=1
            ),

            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        )
    }

    results = []
    best_models = {}

    for model_name, (model, params) in models.items():

        print(f"\n=== GridSearchCV: {model_name} ===")

        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("model", model)
        ])

        # 최적화 이전 교차 검증
        baseline_cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV,
            scoring=["accuracy", "f1", "roc_auc"],
            n_jobs=-1
        )

        # 최적화 이전 모델 학습
        pipeline.fit(X_train, y_train)

        baseline_scores = evaluate_model(
            pipeline, X_test, y_test
        )

        train_accuracy = pipeline.score(
            X_train, y_train
        )

        results.append({
            "Model": model_name,
            "Type": "Before Tuning",
            "CV Accuracy": baseline_cv[
                "test_accuracy"
            ].mean(),
            "CV F1": baseline_cv[
                "test_f1"
            ].mean(),
            "CV F1 Std": baseline_cv[
                "test_f1"
            ].std(),
            "CV ROC-AUC": baseline_cv[
                "test_roc_auc"
            ].mean(),
            "Train Accuracy": train_accuracy,
            "Overfitting Gap": (
                train_accuracy
                - baseline_scores["Accuracy"]
            ),
            **baseline_scores
        })

        # GridSearchCV 실행
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            cv=CV,
            scoring={
                "accuracy": "accuracy",
                "f1": "f1",
                "roc_auc": "roc_auc"
            },
            refit="f1",
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[model_name] = best_model

        print("Best Parameters:")
        print(grid.best_params_)

        print(
            "Best CV F1-score:",
            round(grid.best_score_, 4)
        )

        # 최적화 후 모델 평가
        tuned_scores = evaluate_model(
            best_model, X_test, y_test
        )

        train_accuracy = best_model.score(
            X_train, y_train
        )

        idx = grid.best_index_

        results.append({
            "Model": model_name,
            "Type": "After Tuning",
            "CV Accuracy": grid.cv_results_[
                "mean_test_accuracy"
            ][idx],
            "CV F1": grid.best_score_,
            "CV F1 Std": grid.cv_results_[
                "std_test_f1"
            ][idx],
            "CV ROC-AUC": grid.cv_results_[
                "mean_test_roc_auc"
            ][idx],
            "Train Accuracy": train_accuracy,
            "Overfitting Gap": (
                train_accuracy
                - tuned_scores["Accuracy"]
            ),
            **tuned_scores
        })

        # 모든 탐색 결과 저장
        pd.DataFrame(grid.cv_results_).to_csv(
            f"results/grid_search_"
            f"{model_name.replace(' ', '_')}.csv",
            index=False
        )

        # 최적 하이퍼파라미터 저장
        pd.DataFrame([grid.best_params_]).to_csv(
            f"results/best_params_"
            f"{model_name.replace(' ', '_')}.csv",
            index=False
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/tuning_results.csv",
        index=False
    )

    print("\n=== Hyperparameter Tuning Results ===")

    print(
        results_df.round(4).to_string(index=False)
    )

    # 최적화 전후 과적합 격차 시각화
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Overfitting Gap",
        hue="Type"
    )

    plt.title("Overfitting Gap Before and After Tuning")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Train Accuracy - Test Accuracy")

    plt.tight_layout()

    plt.savefig(
        "results/figures/overfitting_comparison.png",
        dpi=300
    )

    plt.close()

    # 최적화 전후 F1-score 비교
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="F1-score",
        hue="Type"
    )

    plt.title("F1-score Before and After Tuning")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        "results/figures/tuning_f1_comparison.png",
        dpi=300
    )

    plt.close()

    return best_models

# 2. 변수 선택 개수별 교차 검증
def run_feature_count_analysis(X_train, y_train):

    print("\n=== Feature Count Analysis ===")

    feature_counts = [
        3, 5, 10, 15, 20, 25, "All"
    ]

    results = []

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=1
        )
    }

    for count in feature_counts:

        for model_name, model in models.items():

            steps = [
                ("preprocessor", get_preprocessor())
            ]

            if count != "All":

                steps.append((
                    "selector",
                    build_feature_selector(count)
                ))

            steps.append(("model", model))

            pipeline = Pipeline(steps)

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
                "Feature Count": count,
                "Model": model_name,

                "Train Accuracy": scores[
                    "train_accuracy"
                ].mean(),

                "CV Accuracy": scores[
                    "test_accuracy"
                ].mean(),

                "CV F1": scores[
                    "test_f1"
                ].mean(),

                "CV F1 Std": scores[
                    "test_f1"
                ].std(),

                "CV ROC-AUC": scores[
                    "test_roc_auc"
                ].mean(),

                "Overfitting Gap": (
                    scores["train_accuracy"].mean()
                    - scores["test_accuracy"].mean()
                )
            })

            print(
                f"Features: {count}, "
                f"Model: {model_name}, "
                f"CV F1: "
                f"{scores['test_f1'].mean():.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/feature_count_analysis.csv",
        index=False
    )

    print("\n=== Feature Count Results ===")

    print(
        results_df.round(4).to_string(index=False)
    )

    # 변수 개수별 교차 검증 성능 시각화
    plt.figure(figsize=(12, 6))

    for model_name in models:

        subset = results_df[
            results_df["Model"] == model_name
        ]

        x = list(range(len(subset)))

        plt.errorbar(
            x,
            subset["CV F1"],
            yerr=subset["CV F1 Std"],
            marker="o",
            capsize=4,
            label=model_name
        )

    plt.xticks(
        range(len(feature_counts)),
        [str(x) for x in feature_counts]
    )

    plt.xlabel("Number of Selected Features")
    plt.ylabel("Mean CV F1-score")
    plt.title("Feature Count vs Model Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "results/figures/feature_count_comparison.png",
        dpi=300
    )

    plt.close()

    return results_df