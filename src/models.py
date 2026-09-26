"""Baselines and models.

* MajorityBaseline      - always predicts the training late rate (i.e. "everything is on time"
                          at a 0.5 threshold). Sets the floor: ROC-AUC 0.5, PR-AUC = prevalence.
* TypeRateBaseline      - each case gets its type's historical late rate from the training
                          period (smoothed toward the overall rate for rare types).
* logistic_regression   - one-hot categoricals + log-scaled numerics, L2-regularised.
* hist_gradient_boosting- scikit-learn's HistGradientBoostingClassifier with native
                          categorical support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from features import BACKLOG, CATEGORICAL, NUMERIC

SEED = 42


class MajorityBaseline:
    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_)
        return np.column_stack([1 - p, p])


class TypeRateBaseline:
    """Historical late rate of the request type (Bayesian-smoothed with m pseudo-cases)."""

    def __init__(self, m: float = 20.0):
        self.m = m

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.prior_ = float(y.mean())
        g = y.groupby(X["type"])
        self.rates_ = (g.sum() + self.m * self.prior_) / (g.count() + self.m)
        return self

    def predict_proba(self, X):
        p = X["type"].map(self.rates_).fillna(self.prior_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


# ---- logistic regression ----------------------------------------------------
LR_CATEGORICAL = CATEGORICAL + ["hour", "dayofweek", "month"]
LR_NUMERIC = [c for c in NUMERIC if c not in ("hour", "dayofweek", "month")]
_LOG_COLS = ["sla_hours"] + [c for c in BACKLOG if "rate" not in c]


def _log1p_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in _LOG_COLS:
        if c in df:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df


def logistic_regression(C: float = 1.0) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=True), LR_CATEGORICAL),
        ("num", make_pipeline(FunctionTransformer(_log1p_counts, feature_names_out="one-to-one"),
                              SimpleImputer(strategy="median"), StandardScaler()), LR_NUMERIC),
    ])
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


# ---- gradient boosting ------------------------------------------------------
def hist_gradient_boosting(max_iter: int = 400, learning_rate: float = 0.1,
                           max_leaf_nodes: int = 63, min_samples_leaf: int = 100,
                           l2_regularization: float = 1.0, categorical: list[str] | None = None,
                           numeric: list[str] | None = None) -> Pipeline:
    categorical = CATEGORICAL if categorical is None else categorical
    numeric = NUMERIC if numeric is None else numeric
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                encoded_missing_value=np.nan, min_frequency=20), categorical),
         ("num", "passthrough", numeric)],
        verbose_feature_names_out=False,
    )
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        categorical_features=list(range(len(categorical))), early_stopping=False,
        random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
