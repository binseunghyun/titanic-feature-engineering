
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.feature_engineering import create_features

from src.robustness_analysis import (
    get_models,
    get_preprocessor
)

from src.final_analysis import (
    run_shap_analysis,
    run_automl
)


def main():

    df = pd.read_csv("data/train.csv")

    processed_df = create_features(df)

    X = processed_df.drop(columns=["Survived"])
    y = processed_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    best_models = {}

    # 기존 GridSearchCV 최적 설정으로 재학습
    for model_name, model in get_models().items():

        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        best_models[model_name] = pipeline

        print(f"{model_name}: Training Completed")

    # SHAP 분석 재실행
    run_shap_analysis(
        best_models,
        X_train,
        X_test
    )

    # AutoML 비교 실행
    run_automl(
        X_train,
        X_test,
        y_train,
        y_test,
        best_models
    )

    print("\n=== All Final Experiments Completed ===")


if __name__ == "__main__":
    main()
