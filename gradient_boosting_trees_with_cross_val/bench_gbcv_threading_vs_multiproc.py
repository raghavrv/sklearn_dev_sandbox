import numpy as np

from sklearn.ensemble import GradientBoostingRegressorCV, GradientBoostingClassifierCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.datasets import load_boston, fetch_covtype, load_iris, make_classification
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_california_housing

gbccv = GradientBoostingClassifierCV(n_jobs=8, random_state=42)

covtype = fetch_covtype()

X, y = covtype.data[::2], covtype.target[::2]
gbccv.fit(X, y)
