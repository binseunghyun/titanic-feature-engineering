import pandas as pd

from sklearn.model_selection import train_test_split

from src.eda import run_eda
from src.feature_engineering import create_features
from src.model_training import run_experiments

from src.advanced_analysis import (
    run_grid_search,
    run_feature_count_analysis
)

from src.robustness_analysis import (
    run_missing_robustness,
    run_ablation_study
)

from src.final_analysis import (
    run_group_validation,
    run_shap_analysis,
    run_automl
)


def main():

    # STEP 01. 데이터 준비
    df = pd.read_csv("data/train.csv")

    print("=== 데이터 크기 ===")
    print(df.shape)

    print("\n=== 데이터 샘플 ===")
    print(df.head())

    print("\n=== 데이터 기본 정보 ===")
    df.info()

    print("\n=== 결측치 확인 ===")
    print(df.isnull().sum())

    print("\n=== 타겟 변수 분포 ===")
    print(df["Survived"].value_counts())

    # STEP 02. EDA
    run_eda(df)

    # 그룹 검증을 위한 티켓 정보 보존
    groups = df["Ticket"].copy()

    # STEP 03. 파생 변수 생성
    processed_df = create_features(df)

    processed_df.to_csv(
        "results/derived_features.csv",
        index=False
    )

    X = processed_df.drop(columns=["Survived"])
    y = processed_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n=== Train / Test Split ===")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    # STEP 04~05. 기본 실험
    run_experiments(
        X_train, X_test, y_train, y_test
    )

    # STEP 06. 하이퍼파라미터 최적화
    best_models = run_grid_search(
        X_train, X_test, y_train, y_test
    )

    # STEP 07. 변수 선택 개수 최적화
    run_feature_count_analysis(
        X_train, y_train
    )

    # STEP 08. 결측치 강건성
    run_missing_robustness(
        X_train, y_train
    )

    # STEP 09. 파생 변수 기여도
    run_ablation_study(
        X_train, y_train
    )

    # STEP 10. 그룹 기반 검증
    run_group_validation(
        X_train, y_train, groups
    )

    # STEP 11. SHAP 분석
    run_shap_analysis(
        best_models, X_train, X_test
    )

    # STEP 12. AutoML 비교
    run_automl(
        X_train,
        X_test,
        y_train,
        y_test,
        best_models
    )

    print("\n=== All Experiments Completed ===")


if __name__ == "__main__":
    main()
