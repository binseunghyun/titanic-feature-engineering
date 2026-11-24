
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_validate
)

from src.robustness_analysis import (
    get_models,
    get_preprocessor
)

from src.evaluation import evaluate_model


# 1. 그룹 기반 교차 검증
def run_group_validation(X_train, y_train, groups):

    print("\n=== Group-based Cross Validation ===")

    groups = groups.loc[X_train.index]

    methods = {
        "StratifiedKFold": StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),

        "StratifiedGroupKFold": StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
    }

    results = []

    for method_name, cv in methods.items():

        splits = list(
            cv.split(X_train, y_train, groups)
            if method_name == "StratifiedGroupKFold"
            else cv.split(X_train, y_train)
        )

        overlap = []

        for train_idx, val_idx in splits:

            train_groups = set(
                groups.iloc[train_idx]
            )

            val_groups = set(
                groups.iloc[val_idx]
            )

            overlap.append(
                len(train_groups & val_groups)
            )

        for model_name, model in get_models().items():

            pipeline = Pipeline([
                ("preprocessor", get_preprocessor()),
                ("model", model)
            ])

            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=splits,
                scoring=[
                    "accuracy",
                    "f1",
                    "roc_auc"
                ],
                n_jobs=1,
                error_score="raise"
            )

            results.append({
                "Method": method_name,
                "Model": model_name,
                "CV Accuracy":
                    scores["test_accuracy"].mean(),
                "CV F1":
                    scores["test_f1"].mean(),
                "CV F1 Std":
                    scores["test_f1"].std(),
                "CV ROC-AUC":
                    scores["test_roc_auc"].mean(),
                "Mean Group Overlap":
                    np.mean(overlap)
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/group_validation.csv",
        index=False
    )

    print(results_df.round(4).to_string(index=False))

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="CV F1",
        hue="Method"
    )

    plt.title("Standard vs Group-based Validation")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        "results/figures/group_validation.png",
        dpi=300
    )

    plt.close()

    return results_df


# 2. SHAP 설명 가능성 분석
def run_shap_analysis(best_models, X_train, X_test):

    import shap

    print("\n=== SHAP Analysis ===")

    pipeline = best_models["Random Forest"]

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps["model"]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        preprocessor.transform(X_train),
        columns=feature_names
    )

    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=feature_names
    )

    # 모델 학습 시 사용한 특성 개수 확인
    assert X_test_processed.shape[1] == model.n_features_in_

# 트리 내부의 학습 데이터 분포를 활용
    explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent",
    model_output="raw"
    )

    explanation = explainer(
    X_test_processed.to_numpy(dtype=np.float64),
    check_additivity=True
    )

    # 생존 클래스(1)에 대한 SHAP 값
    if explanation.values.ndim == 3:

        class_idx = list(model.classes_).index(1)
        explanation = explanation[:, :, class_idx]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP":
            np.abs(explanation.values).mean(axis=0)
    }).sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )

    importance.to_csv(
        "results/shap_importance.csv",
        index=False
    )

    print(importance.head(15).to_string(index=False))

    # 전체 변수 중요도
    shap.plots.bar(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/shap_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 변수별 예측 영향 방향
    shap.plots.beeswarm(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 승객 한 명에 대한 개별 예측 설명
    shap.plots.waterfall(
        explanation[0],
        max_display=12,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "results/figures/shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP Analysis Completed.")

    return importance


# 3. AutoML 비교
def run_automl(
    X_train,
    X_test,
    y_train,
    y_test,
    best_models
):

    from flaml import AutoML
    from sklearn.model_selection import train_test_split

    print("\n=== AutoML Analysis ===")

    # AutoML 내부 학습 / 검증 분리
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    preprocessor = get_preprocessor()

    X_fit_processed = preprocessor.fit_transform(
        X_fit
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    automl = AutoML()

    settings = {
        "time_budget": 120,
        "metric": "f1",
        "task": "classification",
        "estimator_list": [
            "rf",
            "extra_tree",
            "lrl2"
        ],
        "eval_method": "holdout",
        "retrain_full": False,
        "seed": 42,
        "n_jobs": 1,
        "verbose": 0
    }

    automl.fit(
        X_train=X_fit_processed,
        y_train=y_fit.to_numpy(),
        X_val=X_val_processed,
        y_val=y_val.to_numpy(),
        **settings
    )

    print("Best AutoML Estimator:")
    print(automl.best_estimator)

    print("Best AutoML Configuration:")
    print(automl.best_config)

    # 선택된 설정으로 전체 학습 데이터 재학습
    final_preprocessor = get_preprocessor()

    X_train_processed = (
        final_preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        final_preprocessor.transform(X_test)
    )

    final_automl = AutoML()

    final_automl.fit(
        X_train=X_train_processed,
        y_train=y_train.to_numpy(),
        task="classification",
        metric="f1",
        estimator_list=[automl.best_estimator],
        starting_points={
            automl.best_estimator: automl.best_config
        },
        max_iter=1,
        eval_method="holdout",
        retrain_full=True,
        seed=42,
        n_jobs=1,
        verbose=0
    )

    results = []

    # 기존 최적화 모델 평가
    for model_name, model in best_models.items():

        results.append({
            "Model": model_name,
            **evaluate_model(
                model, X_test, y_test
            )
        })

    # AutoML 모델 평가
    results.append({
        "Model": f"AutoML ({final_automl.best_estimator})",
        **evaluate_model(
            final_automl,
            X_test_processed,
            y_test
        )
    })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/automl_comparison.csv",
        index=False
    )

    print(results_df.round(4).to_string(index=False))

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="F1-score"
    )

    plt.title("Manual Models vs AutoML")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        "results/figures/automl_comparison.png",
        dpi=300
    )

    plt.close()

    return results_df
