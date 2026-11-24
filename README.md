# Titanic Feature Engineering

## 1. 프로젝트 소개

타이타닉 승객 데이터를 활용한 특성 공학 최적화 및 데이터 품질 변화에 따른 머신러닝 모델의 강건성 분석 프로젝트입니다.

단순한 생존 여부 예측을 넘어 전처리 전략, 파생 변수, 변수 선택, 하이퍼파라미터 최적화 및 검증 방식이 모델 성능에 미치는 영향을 비교·분석합니다.

## 2. 데이터셋

- Dataset: Titanic - Machine Learning from Disaster
- Source: https://www.kaggle.com/competitions/titanic/data
- Samples: 891
- Original Columns: 12
- Target: Survived (0: 사망, 1: 생존)
- Task: Binary Classification
- Train/Test Split: 80:20
- Random Seed: 42

## 3. 주요 구현 기능

### 기본 분석

- 탐색적 데이터 분석(EDA)
- IQR 기반 이상치 탐색
- 결측치 처리 전략 3종 비교
- One-Hot 및 Ordinal Encoding 비교
- Standard, MinMax, Robust Scaling 비교
- 파생 변수 6개 생성
- Random Forest 기반 변수 선택
- Logistic Regression 및 Random Forest 비교
- 성능 평가 지표 5종 적용

### 심화 분석

- Scikit-learn Pipeline
- GridSearchCV 기반 하이퍼파라미터 최적화
- 5-Fold Stratified Cross Validation
- 변수 선택 개수별 성능 분석
- 결측치 증가에 따른 모델 강건성 분석
- 파생 변수별 Ablation Study
- 티켓 기반 그룹 교차 검증
- SHAP 설명 가능성 분석
- FLAML 기반 AutoML 비교

## 4. 주요 실험 결과

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8324 | 0.8000 | 0.7536 | 0.7761 | 0.8747 |
| Random Forest (Tuned) | 0.7989 | 0.7538 | 0.7101 | 0.7313 | 0.8456 |
| AutoML | 0.7989 | 0.7391 | 0.7391 | 0.7391 | 0.8503 |

## 5. 프로젝트 차별화 요소

### Missing Data Robustness

검증 데이터에서 기존 나이 관측값의 최대 70%를 추가로 누락시켜 데이터 품질 저하 상황을 재현하였습니다.

실제 나이 결측률이 약 75.42%까지 증가하였음에도 두 모델의 F1-score 감소 폭은 비교적 작게 나타났습니다.

### Ablation Study

파생 변수 6개를 하나씩 제거하여 개별 변수의 성능 기여도를 분석하였습니다.

Logistic Regression에서는 Title, Random Forest에서는 FarePerPerson을 제거했을 때 평균 F1-score 감소 폭이 가장 컸습니다.

### Group-based Cross Validation

동일 티켓을 사용하는 승객이 학습 및 검증 데이터에 동시에 포함되는 문제를 통제하였습니다.

Random Forest의 교차 검증 F1-score는 일반 교차 검증에서 0.7777, 그룹 기반 교차 검증에서 0.7414로 나타났습니다.

### Hyperparameter Optimization

Random Forest의 최대 트리 깊이를 8로 제한하여 학습·테스트 정확도 격차를 18.85%p에서 13.51%p로 감소시켰습니다.

## 6. 실행 방법

### 환경

- Python 3.13
- Windows
- pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- FLAML

### 데이터 준비

Kaggle Titanic 데이터셋에서 train.csv를 다운로드한 후 data/train.csv 경로에 저장합니다.

### 라이브러리 설치

`python -m pip install -r requirements.txt`

### 전체 실험 실행

`python main.py`

### 마지막 분석만 재실행

`python retry_final.py`

## 7. 결과 파일

모든 분석 결과는 results 폴더에 CSV 파일로 저장되며, 시각화 결과는 results/figures 폴더에 PNG 파일로 저장됩니다.

주요 결과 파일:

- experiment_results.csv
- tuning_results.csv
- feature_count_analysis.csv
- missing_robustness_summary.csv
- ablation_results.csv
- group_validation.csv
- shap_importance.csv
- automl_comparison.csv

## 8. 연구의 한계

- 비교적 작은 데이터셋으로 인해 데이터 분할에 따른 성능 변동 가능성이 존재합니다.
- 결측치 강건성 실험은 나이 변수의 무작위 추가 누락 상황을 대상으로 합니다.
- 변수별 기여도는 인과관계를 의미하지 않습니다.
- AutoML은 제한된 탐색 시간과 후보 모델을 사용하므로 일반적인 AutoML 성능을 대표하지 않습니다.
- 동일한 테스트 데이터의 반복 평가 및 하이퍼파라미터 재사용으로 인해 완전히 독립적인 최종 검증에는 한계가 있습니다.

## 9. 결론

특성 공학의 효과는 모델의 학습 방식과 데이터 특성에 따라 다르게 나타났습니다.

본 프로젝트에서는 단순한 정확도 향상뿐만 아니라 과적합, 데이터 품질 저하, 변수별 기여도, 그룹 간 데이터 의존성까지 고려하여 머신러닝 모델의 성능을 종합적으로 분석하였습니다.