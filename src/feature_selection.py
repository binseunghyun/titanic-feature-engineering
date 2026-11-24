from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def build_feature_selector(n_features=10):

    estimator = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=1
    )

    selector = SelectFromModel(
        estimator=estimator,
        threshold=-float("inf"),
        max_features=n_features
    )

    return selector