import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def run_eda(df):
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. 결측치 비율 분석
    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Ratio (%)": df.isnull().mean() * 100
    })

    print("\n=== Missing Value Analysis ===")
    print(missing)

    missing.to_csv("results/missing_values.csv")

    # 결측치 시각화
    plt.figure(figsize=(10, 5))
    missing["Missing Ratio (%)"].plot(kind="bar")
    plt.title("Missing Value Ratio")
    plt.ylabel("Missing Ratio (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "01_missing_values.png", dpi=300)
    plt.close()

    # 2. Histogram: 수치형 변수 분포
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]

    df[numeric_cols].hist(
        figsize=(12, 8),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle("Numerical Feature Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "02_histogram.png", dpi=300)
    plt.close()

    # 3. Boxplot: 이상치 탐색
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, ax in zip(numeric_cols, axes.flatten()):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    plt.savefig(output_dir / "03_boxplot.png", dpi=300)
    plt.close()

    # IQR 기반 이상치 개수 계산
    print("\n=== Outlier Analysis (IQR) ===")

    outlier_results = []

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((df[col] < lower) |
                 (df[col] > upper)).sum()

        print(f"{col}: {count} outliers")

        outlier_results.append({
            "Feature": col,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outlier Count": count
        })

    pd.DataFrame(outlier_results).to_csv(
        "results/outliers.csv",
        index=False
    )

    # 4. Heatmap: 상관관계 분석
    corr = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "04_heatmap.png", dpi=300)
    plt.close()

    # 5. Countplot: 타겟 변수 분포
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Survived")
    plt.title("Survival Distribution")
    plt.xticks([0, 1], ["Not Survived", "Survived"])
    plt.tight_layout()
    plt.savefig(output_dir / "05_target_distribution.png", dpi=300)
    plt.close()

    # 6. 성별에 따른 생존 분포
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Sex", hue="Survived")
    plt.title("Survival by Gender")
    plt.tight_layout()
    plt.savefig(output_dir / "06_survival_by_gender.png", dpi=300)
    plt.close()

    # 7. 객실 등급에 따른 생존 분포
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Survival by Passenger Class")
    plt.tight_layout()
    plt.savefig(output_dir / "07_survival_by_class.png", dpi=300)
    plt.close()

    print("\n=== EDA Completed ===")
    print("Results saved to results/figures/")