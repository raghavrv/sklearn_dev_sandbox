import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_covtype, load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

# dataset = load_digits()
# dataset = load_iris()
dataset = fetch_covtype()
X, y = dataset.data, dataset.target

# Take only 2 classes
# mask = y < 3
# mask = (y == 1) | (y == 2)
# X = X[mask]
# y = y[mask]
# plt.hist(y)
# plt.show()
# X, y = X[::20].copy(), y[::20].copy()
# X, y = X[:100], y[:100]

n_samples, n_features = X.shape

n_estimators = 100
n_jobs = -1

rng = np.random.RandomState(42)

cv = StratifiedShuffleSplit(n_iter=3, test_size=0.3, random_state=rng)

print "The shape of the dataset is %s" % str(X.shape)
print "The number of trees for this benchmarking is %s" % n_estimators

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestClassifier(random_state=0, n_estimators=n_estimators,
                                   missing_values=None, n_jobs=n_jobs)
score = cross_val_score(estimator, X, y, cv=cv).mean()
print "Score with the entire dataset = %.2f" % score

baseline_score = score

scores_missing = []
scores_impute = []

rf_missing = RandomForestClassifier(random_state=0, n_estimators=n_estimators,
                                    missing_values='NaN', n_jobs=n_jobs)
rf_impute = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="median", axis=0)),
                      ("forest", RandomForestClassifier(
                                         random_state=0,
                                         n_estimators=n_estimators,
                                         n_jobs=n_jobs))])

missing_fraction_range = []
missing_mask = np.zeros(X.shape, dtype=bool)

for _ in range(40):
    X_missing = X.copy()
    X_missing_feat_min = X.copy()
    rv = rng.randn(*X.shape)
    thresh = np.sort(rv.ravel())[int(0.05 * n_samples * n_features)]
    missing_mask += rv < thresh
    missing_mask[y!=1] = False  # Features should go missing only for y=1
    missing_fraction = np.mean(missing_mask)
    missing_fraction_range.append(missing_fraction)
    X_missing[missing_mask] = np.nan

    train, test = iter(cv.split(X, y)).next()
    # print(len(train), len(test))
    # score_missing = rf_missing.fit(X_missing[train], y[train]).score(X[test], y[test])
    # score_impute = rf_impute.fit(X_missing[train], y[train]).score(X[test], y[test])
    score_missing = cross_val_score(rf_missing, X_missing, y, cv=cv).mean()
    score_impute = cross_val_score(rf_impute, X_missing, y, cv=cv).mean()
    scores_missing.append(score_missing)
    scores_impute.append(score_impute)
    print ("Score RF with the %s %% missing = %.2f"
           % (missing_fraction*100, score_missing))
    print ("Score RF+Imp. with the %s %% missing = %.2f"
           % (missing_fraction*100, score_impute))
    # print "The missing mask is \n", missing_mask

np.save('scores_missing.npy', scores_missing)
np.save('scores_impute.npy', scores_impute)
np.save('missing_fraction_range.npy', missing_fraction_range)
# plt.close('all')
# plt.plot(missing_fraction_range, scores_missing, 'o--', color='r', label='RF mv')
# plt.plot(missing_fraction_range, scores_impute, 'o--', color='b', label='RF imp.')
# plt.axhline(baseline_score, label='no missing', color='k')
# plt.xlabel('Missing fraction')
# plt.ylabel('Score')
# plt.legend(loc='lower left')
# plt.show()
