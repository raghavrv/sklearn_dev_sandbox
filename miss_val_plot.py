import numpy as np
import matplotlib.pyplot as plt

baseline_score = np.load('baseline_score.npy')
missing_fraction_range = np.load('missing_fraction_range.npy')
scores_missing = np.load('scores_missing.npy')
scores_impute = np.load('scores_impute.npy')

plt.close('all')
plt.plot(missing_fraction_range, scores_missing, 'o--', color='r', label='RF mv')
plt.plot(missing_fraction_range, scores_impute, 'o--', color='b', label='RF imp.')
plt.axhline(baseline_score, label='no missing', color='k')
plt.xlabel('Missing fraction')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()
