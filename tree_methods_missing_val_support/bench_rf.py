
# coding: utf-8

# #### Imports

# In[1]:

import time

from collections import OrderedDict
from functools import partial

import numpy as np
import scipy as sp
from scipy.stats import rankdata

from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
# NOTE The add_indicator_feature is not in sklearn 0.18.0 This is a patched version for benchmarking...
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.io.arff import loadarff as scipy_loadarff
from arff import load as liac_loadarff

# https://github.com/raghavrv/pyarff
# NOTE pyarff is very much experimental and is not finished fully yet
# scipy arff is a bit slow but doesn't encode categories.
# LIAC is another arff reader that works on sparse too but sometimes breaks with
# datasets that surround data with quotes
import pyarff

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier

import matplotlib.image as img
import matplotlib.pyplot as plt

import pygraphviz as pgv

from sklearn.tree import export_graphviz
from io import BytesIO


def get_graph(dtc, n_classes, feat_names=None, size=[7, 7]):
    # Get the dot graph of our decision tree
    tree_dot = export_graphviz(
        dtc, out_file=None, feature_names=feat_names, rounded=True, filled=True,
        special_characters=True, class_names=list(map(str, range(n_classes))), max_depth=10)
    # Convert this dot graph into an image
    g = pgv.AGraph(tree_dot)
    g.layout('dot')
    g.draw(path='temp.png')
    # Plot it
    plt.figure().set_size_inches(*size)
    plt.axis('off')
    plt.imshow(img.imread(fname='temp.png'))
    plt.show()

rng = np.random.RandomState(0)

all_datasets = OrderedDict()

def print_progress_bar(n):
    n = int(n)
    print("\r" + "█" + ("█" *  n) + ("-" * (100 - n)) + "█ %d%%" % n,
          end="" if n != 100 else "\n", flush=True)


# ### Let's do some benchmarks on real world datasets that have missing values

# In[4]:

# Anneal dataset - http://www.openml.org/d/2

meta1, data1 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/anneal.arff')
data1 = data1[:-1]  # The last one seems to have a wrong class label. Check pyarff.
X, y = data1[:, :-1], data1[:, -1]
all_datasets['anneal'] = (X, y)

np.bincount(y.astype(int))


# In[5]:

# KDDCUP09_churn - http://www.openml.org/d/1112

# start = time.time()
# data = scpy_arff_load('/home/raghavrv/code/datasets/arff/KDDCup09_churn.arff')[0]
# print("KDDCUP09_churn ARFF dataset loaded in %0.8fs" % (time.time() - start))
# X = np.array([np.array(list(data_i))[:190] for data_i in data]).astype(float)
# y = np.array([data_i[-1] for data_i in data]).astype(int)

meta2, data2 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/KDDCup09_churn.arff',
                                        encode_nominals=True)
X, y = data2[:, :-1], data2[:, -1]
all_datasets['KDDCUP09_churn'] = (X, y)


# In[6]:

# KDDCUP09_appetency - http://www.openml.org/d/1111

meta3, data3 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/KDDCup09_appetency.arff',
                                        encode_nominals=True)
X, y = data3[:, :-1], data3[:, -1]
all_datasets['KDDCUP09_appetency'] = (X, y)


# In[7]:

# KDDCUP09_upselling - http://www.openml.org/d/1114

meta4, data4 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/KDDCup09_upselling.arff',
                                        encode_nominals=True)
X, y = data4[:, :-1], data4[:, -1]
all_datasets['KDDCUP09_upselling'] = (X, y)


# In[8]:

# CSJ - http://www.openml.org/d/23380

meta5, data5 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/cjs.arff',
                                        encode_nominals=True)

target_index = meta5['attributes'][b'TR']['order']
X = np.hstack((data5[:, :target_index], data5[:, target_index+1:])).astype(float)
y = data5[:, target_index].astype(int)
all_datasets['cjs'] = (X, y)

np.bincount(y)


# In[9]:

# Soy Bean dataset - http://www.openml.org/d/42

meta6, data6 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/soybean.arff',
                                        encode_nominals=True)
X, y = data6[:, :-1].astype(float), data6[:, -1].astype(int)
y[-1] = 2  # The last label seems to not be loaded properly (Pyarff bug)
all_datasets['soy_bean'] = (X, y)

np.bincount(y)


# In[10]:

# Adult Census dataset - https://archive.ics.uci.edu/ml/datasets/Adult

meta, data = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/adult-census.arff',
                                      encode_nominals=True)
X, y = data[:, :-1].astype(float), data[:, -1].astype(int)
all_datasets['adult_census'] = (X, y)

np.bincount(y)


# In[11]:

# Lymphoma 2 classes - http://www.openml.org/d/1101

meta7, data7 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/lymphoma_2classes.arff',
                                        encode_nominals=True)
X, y = data7[:, :-1].astype(float), data7[:, -1].astype(int)
all_datasets['lymphoma_2classes'] = (X, y)

np.bincount(y)


# In[12]:

# Lymphoma 9 classes - http://www.openml.org/d/1102


meta8, data8 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/lymphoma_9classes.arff',
                                        encode_nominals=True)
X, y = data8[:, :-1].astype(float), data8[:, -1].astype(int)
all_datasets['lymphoma_9classes'] = (X, y)

np.bincount(y)


# In[13]:

# KDD98 - http://www.openml.org/d/23513

meta9, data9 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/kdd98_data.arff',
                                        encode_nominals=True)
target_index = meta9['attribute_names_in_order'].index(b'TARGET_B')
X = np.hstack([data9[:, :target_index], data9[:, target_index+1:]]).astype(float)
y = data9[:, target_index].astype(int)
all_datasets['kdd98'] = (X, y)

np.bincount(y)


# In[14]:

# Colleges US News binarized - http://www.openml.org/d/930

# data10_ = scipy_loadarff('/home/raghavrv/code/datasets/arff/colleges_usnews.arff')[0]
# X_ = np.array([list(data_i) for data_i in data10_])[:-1]
# y_ = np.array([data_i[-1] for data_i in data10_])[:-1]

meta10, data10 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/colleges_usnews.arff',
                                          encode_nominals=True)
data10 = data10[:-1]  # The last one seems to have a wrong class label. Bug in pyarff.
X, y = data10[:, :-1].astype(float), data10[:, -1].astype(int)
all_datasets['colleges_usnews'] = (X, y)

np.bincount(y)


# In[15]:

# arrhythmia - http://www.openml.org/d/5

meta11, data11 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/dataset_5_arrhythmia.arff',
                                          encode_nominals=True)

target_index = meta11['attributes'][b'class']['order']
X = np.hstack((data11[:, :target_index], data11[:, target_index+1:])).astype(float)
y = data11[:, target_index].astype(int)
all_datasets['arrhythmia'] = (X, y)

np.bincount(y)


# In[16]:

# Vote - http://www.openml.org/d/56

meta12, data12 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/vote.arff',
                                          encode_nominals=True)
target_index = meta12['attributes'][b'Class']['order']
X = np.hstack((data12[:, :target_index], data12[:, target_index+1:])).astype(float)
y = data12[:, target_index].astype(int)
all_datasets['vote'] = (X, y)

np.bincount(y)


# In[17]:

# Pro Football Scores - http://www.openml.org/d/470

meta13, data13 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/profb.arff',
                                          encode_nominals=True)
target_index = meta13['attributes'][b'Home/Away']['order']
X = np.hstack((data13[:, :target_index], data13[:, target_index+1:])).astype(float)
y = data13[:, target_index].astype(int)
all_datasets['pro football scores'] = (X, y)

np.bincount(y)


# In[18]:

# Mice Protein - http://www.openml.org/d/4550

meta14, data14 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/mice.arff',
                                          encode_nominals=True)
target_index = meta14['attributes'][b'class']['order']
X = np.hstack((data14[:, :target_index], data14[:, target_index+1:])).astype(float)
y = data14[:, target_index].astype(int)
all_datasets['mice_protein'] = (X, y)

np.bincount(y)


# In[19]:

# IPUMS98 small - http://www.openml.org/d/381

meta14, data14 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/ipums_la_98-small.arff',
                                          encode_nominals=True)
target_index = meta14['attributes'][b'movedin']['order']
X = np.hstack((data14[:, :target_index], data14[:, target_index+1:])).astype(float)
y = data14[:, target_index].astype(int)
all_datasets['ipums_98'] = (X, y)


# In[20]:

# IPUMS97 small - http://www.openml.org/d/381

meta, data = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/ipums_la_97-small.arff',
                                      encode_nominals=True)
target_index = meta['attributes'][b'movedin']['order']
X = np.hstack((data[:, :target_index], data[:, target_index+1:])).astype(float)
y = data[:, target_index].astype(int)
all_datasets['ipums_97'] = (X, y)

class_count = np.bincount(y)
classes = np.unique(y)
print(list(zip(classes, class_count[classes.tolist()])))


# In[21]:

# IPUMS99 small - http://www.openml.org/d/381

meta16, data16 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/arff/ipums_la_99-small.arff',
                                          encode_nominals=True)
target_index16 = meta16['attributes'][b'movedin']['order']
X16 = np.hstack((data16[:, :target_index16], data16[:, target_index16+1:])).astype(float)
y16 = data16[:, target_index16].astype(int)
all_datasets['ipums_99'] = (X16, y16)

class_count16 = np.bincount(y16)
classes16 = np.unique(y16)
print(list(zip(classes16, class_count16[classes16.tolist()])))


# In[22]:

# 19) Census Income Dataset (Large) - http://sci2s.ugr.es/keel/dataset.php?cod=195

meta17, data17 = pyarff.load_arff_dataset('/home/raghavrv/code/datasets/more_missing_datasets/census.arff',
                                          encode_nominals=True)
target_index17 = meta17['attributes'][b'Class']['order']
X17 = np.hstack((data17[:, :target_index17], data17[:, target_index17+1:])).astype(float)
y17 = data17[:, target_index17].astype(int)
all_datasets['census_income_large'] = (X17, y17)

class_count17 = np.bincount(y17)
classes17 = np.unique(y17)
print(list(zip(classes17, class_count17[classes17.tolist()])))


# In[ ]:

# 20) Higgs Boson challenge Kaggle

from sklearn.preprocessing import LabelEncoder
X_train = np.loadtxt('/home/raghavrv/code/datasets/kaggle_higgs_challenge/training.csv',
                     delimiter=',', unpack=True, skiprows=1, usecols=np.arange(1, 32)).T
X_test = np.loadtxt('/home/raghavrv/code/datasets/kaggle_higgs_challenge/test.csv',
                     delimiter=',', unpack=True, skiprows=1, usecols=np.arange(1, 31)).T
y_train = np.loadtxt('/home/raghavrv/code/datasets/kaggle_higgs_challenge/training.csv',
                  delimiter=',', unpack=True, skiprows=1, usecols=(32,), dtype=bytes)

# Higgs Boson dataset represents unavailable data as -999
X_train[X_train==-999.] = np.nan
X_test[X_test==-999.] = np.nan

sample_weight = X_train[:, -1]

X_train = X_train[:, :-1]

all_datasets['higgs_boson'] = (X_train, y_train, sample_weight)

class_count = np.bincount((y_train == b's').astype(int))
classes = np.unique(y_train)
print(list(zip(classes, class_count[(classes == b's').astype(int).tolist()])))


# ### Benchmark and Summarize

# In[ ]:

import warnings
from itertools import product
warnings.filterwarnings("ignore")

n_jobs = 8
n_estimators = 50
max_depth = None
cv=StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=0)

verbose=True

print('n_jobs=%d'%n_jobs)

all_results = OrderedDict()

for n_estimators, max_depth, bootstrap in product((10, 50, 100), (10, 20, 30, None), (True, False)):
    print()
    print("=*" * 50)
    print("n_estimators=%d; max_depth=%s; bootstrap=%s" % (n_estimators, str(max_depth), str(bootstrap)))
    print("=*" * 50)
    print()

    all_benchmarks = OrderedDict()
    all_score_ranks = OrderedDict()
    all_time_ranks = OrderedDict()
    all_estimators = OrderedDict()

    for dataset_desc, data in all_datasets.items():
        if len(data) == 2:
            sw = ""
            X, y, sample_weight = data[0], data[1], None
        else:
            sw = "\nTraining with sample weights."
            X, y, sample_weight = data

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = np.unique(y).shape[0]
        benchmarks = []  # Tuple of (<technique>, mean cross val score, total fit time)
        estimators = []

        missing_mask = np.isnan(X)
        missing_samples, missing_features = np.where(missing_mask)
        if verbose:
            print("Dataset %s with %d samples and %d features. n_classes=%d; It has %0.4f%% values missing;%s" 
                  % (dataset_desc, n_samples, n_features, n_classes,
                     100. * (np.sum(missing_mask) / (n_samples * n_features)), sw))
            print("====================================================================================================\n")
        

            print_progress_bar(0)

        # 1) Estimate score with RFC natively handling missing values
        estimator = RandomForestClassifier(random_state=0, n_jobs=n_jobs, bootstrap=bootstrap,
                                           missing_values="NaN", n_estimators=n_estimators,
                                           max_depth=max_depth)
        scores = cross_val_score(estimator, X, y,  fit_params={'sample_weight':sample_weight}, cv=cv)
        score, score_err = scores.mean(), sp.stats.sem(scores)

        if verbose: print_progress_bar(18)


        start = time.time()
        estimator.fit(X, y, sample_weight=sample_weight)
        estimators.append(estimator)
        benchmarks.append(('Random Forest natively handling it', scores, time.time() - start))

        if verbose: print_progress_bar(20)



        # 2) Estimate score after mean imputation of missing values without indicator features
        estimator = Pipeline([("Impute", Imputer(missing_values="NaN", strategy="mean",
                                                 add_indicator_features=False, axis=0)),
                              ("rf", RandomForestClassifier(random_state=0, bootstrap=bootstrap,
                                                            n_jobs=n_jobs, n_estimators=n_estimators,
                                                            max_depth=max_depth))])
        scores = cross_val_score(estimator, X, y, fit_params={'rf__sample_weight':sample_weight}, cv=cv)
        score, score_err = scores.mean(), sp.stats.sem(scores)

        if verbose: print_progress_bar(38)

        start = time.time()
        estimator.fit(X, y, rf__sample_weight=sample_weight)
        estimators.append(estimator)
        benchmarks.append(('mean imputation of the missing values', scores, time.time() - start))

        if verbose: print_progress_bar(40)



        # 3) Estimate score after mean imputation of the missing values with indicator matrix
        estimator = Pipeline([("Impute", Imputer(missing_values="NaN", strategy="mean",
                                                 add_indicator_features=True, axis=0)),
                              ("rf", RandomForestClassifier(random_state=0, bootstrap=bootstrap,
                                                             n_jobs=n_jobs, n_estimators=n_estimators,
                                                             max_depth=max_depth))])
        scores = cross_val_score(estimator, X, y, fit_params={'rf__sample_weight':sample_weight}, cv=cv)
        score, score_err = scores.mean(), sp.stats.sem(scores)

        if verbose: print_progress_bar(58)

        start = time.time()
        estimator.fit(X, y, rf__sample_weight=sample_weight)
        estimators.append(estimator)
        benchmarks.append(('mean imputation of the missing values w/indicator features', scores, time.time() - start))

        if verbose: print_progress_bar(60)




        # 4) Estimate score after median imputation of missing values without indicator features
        estimator = Pipeline([("Impute", Imputer(missing_values="NaN", strategy="median",
                                                 add_indicator_features=False, axis=0)),
                              ("rf", RandomForestClassifier(random_state=0, bootstrap=bootstrap,
                                                             n_jobs=n_jobs, n_estimators=n_estimators,
                                                             max_depth=max_depth))])
        scores = cross_val_score(estimator, X, y, fit_params={'rf__sample_weight':sample_weight}, cv=cv)
        score, score_err = scores.mean(), sp.stats.sem(scores)

        if verbose: print_progress_bar(78)

        start = time.time()
        estimator.fit(X, y, rf__sample_weight=sample_weight)
        estimators.append(estimator)
        benchmarks.append(('median imputation of the missing values', scores, time.time() - start))

        if verbose: print_progress_bar(80)


        # 5) Estimate score after median imputation of the missing values with indicator matrix
        estimator = Pipeline([("Impute", Imputer(missing_values="NaN", strategy="median",
                                                 add_indicator_features=True, axis=0)),
                              ("rf", RandomForestClassifier(random_state=0, bootstrap=bootstrap,
                                                            n_jobs=n_jobs, n_estimators=n_estimators,
                                                            max_depth=max_depth))])
        scores = cross_val_score(estimator, X, y, fit_params={'rf__sample_weight':sample_weight}, cv=cv)
        score, score_err = scores.mean(), sp.stats.sem(scores)

        if verbose: print_progress_bar(98)

        start = time.time()
        estimator.fit(X, y, rf__sample_weight=sample_weight)
        estimators.append(estimator)
        benchmarks.append(('median imputation of the missing values w/indicator features',
                           scores, time.time() - start))


        if verbose: print_progress_bar(100)


        names, scores, times = list(zip(*benchmarks))

        scores = np.array(scores)
        ## Per estimator/technique, compute the ranks based on the score per fold 
        #ranks = np.array(list(rankdata(scores[:, i], method='min') for i in range(scores.shape[1]))).T
        #ranks_score = rankdata(ranks.mean(axis=1), method='min')

        scores = scores.mean(axis=1)
        ranks_score = rankdata(-scores, method='min')
        #print(ranks_score)
        ranks_fit_time = rankdata(times, method='min')

        all_benchmarks[dataset_desc] = benchmarks
        all_estimators[dataset_desc] = estimators
        all_score_ranks[dataset_desc] = ranks_score
        all_time_ranks[dataset_desc] = ranks_fit_time

        if verbose:
            # Print statistics for this dataset
            for i, benchmark in enumerate(benchmarks):
                print("%s Got a score of %0.8f [%s] with %s (Train time %0.2fs [%s] %s)"
                      % ("*" if ranks_score[i] == 1 else " ",
                         benchmark[1].mean(), ranks_score[i], benchmark[0],
                         benchmark[2], ranks_fit_time[i],
                         "*" if ranks_fit_time[i] == 1 else " "))
        
            
    # After all the datasets print the summary for one set of parameters
    print("Benchmark summary")
    print("=================")
    if verbose:
        print("n_estimators=%d; max_depth=%s; bootstrap=%s" % (n_estimators, str(max_depth), str(bootstrap)))


    methods = list(zip(*next(iter(all_benchmarks.values()))))[0]
    n_methods = len(methods)

    n_datsets = len(all_benchmarks)
    rank_suffix = [None, 'st', 'nd', 'rd', 'th', 'th']

    if verbose:
        print("\n%d datasets were tested.\n\nn_estimators=%d and \ncv=%s" 
              % (n_datsets, n_estimators, str(cv)))
        print("\n\n")


    score_ranks_per_method = list(zip(*list(all_score_ranks.values())))
    _bincount_ = partial(np.bincount, minlength=n_methods + 1)
    bincount_score_ranks_per_method = list(map(_bincount_, score_ranks_per_method))
    fittime_ranks_per_method = list(zip(*list(all_time_ranks.values())))
    bincount_fittime_ranks_per_method = list(map(_bincount_, fittime_ranks_per_method))

    if verbose:
        print('-' * 100)
        for i, method in enumerate(methods):
            print(method, '\n')

            score_rank_counts = bincount_score_ranks_per_method[i][1:]
            time_rank_counts = bincount_fittime_ranks_per_method[i][1:]

            rank_stats = "".join(("%d%s (%d / %d times) "
                                  %  (r, rank_suffix[r], score_rank_counts[r-1], n_datsets))
                                  for r in range(1, 6))
            print("--> got ranked by score \n - %s" % (rank_stats))
            print()
            rank_stats = "".join(("%d%s (%d / %d times) "
                                  %  (r, rank_suffix[r], time_rank_counts[r-1], n_datsets))
                                  for r in range(1, 6))
            print("--> got ranked by lower fit-time on entire dataset \n - %s" % (rank_stats))
            print()

            print('-' * 100)
    
    
    # Plot histogram of ranks

    # ind = np.array([1, 1.3, 1.6, 1.9, 2.2])  # the x locations for the groups
    ind = np.array([1, 2.5, 4, 5.5, 7])
    width = 0.2       # the width of the bars

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 15)

    colors = ['r', 'g', 'b', 'k', 'y']

    rects1, rects2 = [], []

    for i, method in enumerate(methods):
        this_scores = bincount_score_ranks_per_method[i][1:]
        this_times = bincount_fittime_ranks_per_method[i][1:]
        score_rank_counts = np.hstack((np.cumsum(this_scores[:-1]), this_scores[-1]))
        time_rank_counts = np.hstack((np.cumsum(this_times[:-1]), this_times[-1]))

        #  ((width + 0.1) * 5 + 0.5) * i
        rects1.append(ax[0].bar(ind + width * i,
                                score_rank_counts, width, color=colors[i]))
        rects2.append(ax[1].bar(ind + width * i,
                                time_rank_counts, width, color=colors[i]))


    # add some text for labels, title and axes ticks
    ax[0].set_ylabel('Number of Datasets')
    ax[0].set_xlabel('(Bins of Ranks)')

    ax[0].set_title('Score rank statistics\n(Higher score ranks lower)\n')
    ax[0].set_xticks(ind + width * 2.5)
    # ax[0].set_xticklabels(('1', '2', '3', '4', '5'))
    ax[0].set_xticklabels(('Ranked Best', 'Ranked 1st or 2nd', '1st 2nd or 3rd', '1st 2nd 3rd or 4th', 'Ranked Worst'))
    ax[0].set_ylim((-0.5, n_datsets + 1))

    ax[1].set_title('Fit time rank statistics\n(faster ranks lower)\n')
    ax[1].set_xlabel('(Bins of Ranks)')
    ax[1].set_xticks(ind + width * 2.5)
    ax[1].set_xticklabels(('Ranked Best', 'Ranked 1st or 2nd', '1st 2nd or 3rd', '1st 2nd 3rd or 4th', 'Ranked Worst'))
    ax[1].set_ylim((-0.5, n_datsets + 1))


    def autolabel(rects, ax):
        # attach some text labels
        for bars in rects:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        '%d' % int(height),
                        ha='center', va='bottom')

    autolabel(rects1, ax[0])
    autolabel(rects2, ax[1])

    ax[1].legend(rects1, methods, bbox_to_anchor=(0.1, 2.7), loc='upper left')

    #ax[1].title("Comparison of performance of various methods of handling missing values.\nn_estimators=10, mean across 20 Stratified Shuffle Splits with 0.2 test size")

    plt.savefig('n_estimators_%d__max_depth_%s__bootstrap_%s.png' % (n_estimators, str(max_depth), str(bootstrap)))
    # plt.show()
    print('=' * 100)
    print()
    
    all_results[(n_estimators, max_depth, bootstrap)] = (all_benchmarks, all_score_ranks,
                                                         all_time_ranks, all_estimators)

