import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import (
    build_preprocessor,
    EXPERIMENTS
)

from src.feature_selection import build_feature_selector
from src.evaluation import evaluate_model, save_results


def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }


def run_experiments(X_train, X_test, y_train, y_test):

    results = []
    selected_features = []

    # Base 모델에 사용할 원본 수치형 변수
    base_columns = [
        "Pclass",
        "SibSp",
        "Parch",
        "Fare"
    ]

    # 1. Base 모델 학습
    for model_name, model in get_models().items():

        model.fit(
            X_train[base_columns],
            y_train
        )

        scores = evaluate_model(
            model,
            X_test[base_columns],
            y_test
        )

        results.append({
            "Experiment": "Base",
            "Model": model_name,
            "Imputation": "None",
            "Encoding": "None",
            "Scaling": "None",
            "Selection": False,
            "Derived Features": False,
            "Feature Count": len(base_columns),

            "Train Accuracy": model.score(
                X_train[base_columns], y_train
            ),

            **scores
        })

    # 추가 비교 실험
    experiments = {
        name: config.copy()
        for name, config in EXPERIMENTS.items()
    }

    # 파생 변수 생성 효과 비교
    experiments["Exp-6"] = {
        **experiments["Exp-4"],
        "use_derived": False
    }

    # 인코딩 효과 비교
    experiments["Exp-7"] = {
        **experiments["Exp-4"],
        "encoding": "label"
    }

    # MinMax 스케일링 효과 비교
    experiments["Exp-8"] = {
        **experiments["Exp-4"],
        "scaling": "minmax"
    }

    # Robust 스케일링 효과 비교
    experiments["Exp-9"] = {
        **experiments["Exp-4"],
        "scaling": "robust"
    }

    # 최빈값 결측치 처리 효과 비교
    experiments["Exp-10"] = {
        **experiments["Exp-4"],
        "imputation": "most_frequent"
    }

    # 2. 전처리 실험
    for experiment, config in experiments.items():

        for model_name, model in get_models().items():

            print(f"\nTraining {experiment}: {model_name}")

            use_derived = config.get(
                "use_derived", True
            )

            preprocessor = build_preprocessor(
                imputation=config["imputation"],
                encoding=config["encoding"],
                scaling=config["scaling"],
                use_derived=use_derived
            )

            steps = [
                ("preprocessor", preprocessor)
            ]

            # 변수 선택 적용
            if config["selection"]:

                steps.append((
                    "selector",
                    build_feature_selector(10)
                ))

            steps.append(("model", model))

            # 전처리 + 변수 선택 + 모델 학습 통합
            pipeline = Pipeline(steps)

            pipeline.fit(X_train, y_train)

            # 변환된 특성 이름 확인
            feature_names = pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()

            if config["selection"]:

                mask = pipeline.named_steps[
                    "selector"
                ].get_support()

                feature_names = feature_names[mask]

                print("Selected Features:")
                print(feature_names)

                for feature in feature_names:

                    selected_features.append({
                        "Experiment": experiment,
                        "Model": model_name,
                        "Feature": feature
                    })

            # 성능 평가
            scores = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            results.append({
                "Experiment": experiment,
                "Model": model_name,
                "Imputation": config["imputation"],
                "Encoding": config["encoding"],
                "Scaling": config["scaling"],
                "Selection": config["selection"],
                "Derived Features": use_derived,
                "Feature Count": len(feature_names),

                "Train Accuracy": pipeline.score(
                    X_train, y_train
                ),

                **scores
            })

            print(
                f"Accuracy: {scores['Accuracy']:.4f}"
            )

            print(
                f"F1-score: {scores['F1-score']:.4f}"
            )

            print(
                f"ROC-AUC: {scores['ROC-AUC']:.4f}"
            )

            # Random Forest 변수 중요도 시각화
            if model_name == "Random Forest":

                importance = pd.DataFrame({
                    "Feature": feature_names,

                    "Importance": pipeline.named_steps[
                        "model"
                    ].feature_importances_
                })

                importance = importance.sort_values(
                    "Importance",
                    ascending=False
                )

                importance.to_csv(
                    f"results/{experiment}_importance.csv",
                    index=False
                )

                plt.figure(figsize=(10, 7))

                sns.barplot(
                    data=importance.head(15),
                    x="Importance",
                    y="Feature"
                )

                plt.title(
                    f"{experiment} Feature Importance"
                )

                plt.tight_layout()

                plt.savefig(
                    f"results/figures/"
                    f"{experiment}_importance.png",
                    dpi=300
                )

                plt.close()

    # 선택된 변수 저장
    pd.DataFrame(selected_features).to_csv(
        "results/selected_features.csv",
        index=False
    )

    # 전체 실험 결과 저장
    results_df = save_results(results)

    # F1-score 기준 최고 성능 실험
    best = results_df.sort_values(
        "F1-score",
        ascending=False
    ).iloc[0]

    print("\n=== Highest Test F1-score ===")
    print(best)

    return results_df