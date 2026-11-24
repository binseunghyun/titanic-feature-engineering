import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


# 1. 파생 변수 생성
def create_features(df):

    df = df.copy()

    # 전체 가족 인원
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 혼자 탑승했는지 여부
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 가족 인원 기준 환산 요금
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # 연령대 구분
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=[
            "Child",
            "Teenager",
            "Adult",
            "MiddleAge",
            "Senior"
        ],
        include_lowest=True
    ).astype(object)

    # 이름에서 호칭 추출
    df["Title"] = df["Name"].str.extract(
        r",\s*([^.]*)\.",
        expand=False
    ).str.strip()

    # 동일한 의미의 호칭 통합
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    # 희귀 호칭 통합
    common_titles = ["Mr", "Mrs", "Miss", "Master"]

    df["Title"] = df["Title"].where(
        df["Title"].isin(common_titles),
        "Rare"
    )

    # 객실 정보 존재 여부
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # 불필요한 원본 컬럼 제거
    df = df.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"]
    )

    print("\n=== Derived Features ===")

    print(df[[
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "AgeGroup",
        "Title",
        "HasCabin"
    ]].head(10))

    return df


# 2. 전처리 파이프라인 생성
def build_preprocessor(
    imputation="mean",
    encoding="onehot",
    scaling="standard",
    use_derived=True
):

    numeric_features = [
        "Age",
        "Fare",
        "SibSp",
        "Parch"
    ]

    categorical_features = [
        "Pclass",
        "Sex",
        "Embarked"
    ]

    if use_derived:
        numeric_features += [
            "FamilySize",
            "FarePerPerson"
        ]

        categorical_features += [
            "AgeGroup",
            "Title",
            "HasCabin",
            "IsAlone"
        ]

    # 인코딩 방식 선택
    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    elif encoding == "label":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    else:
        raise ValueError("Invalid encoding method")

    # 스케일링 방식 선택
    if scaling == "standard":
        scaler = StandardScaler()

    elif scaling == "minmax":
        scaler = MinMaxScaler()

    elif scaling == "robust":
        scaler = RobustScaler()

    elif scaling == "none":
        scaler = "passthrough"

    else:
        raise ValueError("Invalid scaling method")

    # 수치형 변수 처리
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputation)),
        ("scaler", scaler)
    ])

    # 범주형 변수 처리
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", encoder)
    ])

    # 수치형과 범주형 전처리 통합
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# 3. 실험별 전처리 설정
EXPERIMENTS = {
    "Exp-1": {
        "imputation": "mean",
        "encoding": "onehot",
        "scaling": "standard",
        "selection": False
    },

    "Exp-2": {
        "imputation": "median",
        "encoding": "label",
        "scaling": "minmax",
        "selection": True
    },

    "Exp-3": {
        "imputation": "most_frequent",
        "encoding": "onehot",
        "scaling": "robust",
        "selection": True
    },

    "Exp-4": {
        "imputation": "median",
        "encoding": "onehot",
        "scaling": "standard",
        "selection": False
    },

    "Exp-5": {
        "imputation": "median",
        "encoding": "onehot",
        "scaling": "standard",
        "selection": True
    }
}