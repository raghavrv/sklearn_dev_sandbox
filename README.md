##### Notes

* The PR - [scikit-learn/scikit-learn #5974](https://github.com/scikit-learn/scikit-learn/pull/5974)
* The missing values (in each split) are handled by deciding the best way to send them based on the impurtiy decrease.
* XGBoost also seems to do this our way (https://github.com/dmlc/xgboost/issues/21#issuecomment-51982962) (Thanks to Jacob!)
* This is working for `RandomForestClassifier`, `DecisionTreeClassifier`, `BestSplitter` (dense only) and `ClassificationCriterion`.
* Yet to implement the this for `BestFirstTreeBuilder`, `RandomSplitter`, and `{Best|Random}SparseSplitter` (waiting for benchmarking the above and getting convincing results)
* Yet to implement the this for `RandomForestRegressor`, `DecisionTreeRegressor` and `RegressionCriterion`
* From the bench (on covtype/mnist with induced missing and census income dataset/adult with inherent missing), what we know
    * That our implementation is performing either better, or almost as same as or sometimes slightly worse than imputation when the missing is correlated with the labels
    * When the missing is completely at random imputation blows our method out of the water.
    * (Personal woe - Imputation is evil :sob:)

<hr>

##### Some recent benchmarks

The benchmark notebook - https://github.com/rvraghav93/miss_val_bench/blob/master/missing_val_bench.ipynb

**Dataset:** One tenth of the covtype dataset <br>
**Scoring:** mean score across 3 iterations of `StratifiedShuffleSplit` <br>
**n_estimators:** 50 <br>
1. When all the classes are present, and missing values across all the features correspond to one of the classes. (MNAR) (class 2) (1/2 of covtype)

![](https://i.imgur.com/T2cDztI.png)

2. When the missing values are completely at random (MCAR).  (1/2 of covtype) (This PR's implementation tries to extract information out of randomness and hence is expected to perform badly  for MCAR.)

![](https://i.imgur.com/6eHuep7.png)

**XGBoostClassifier**
**Dataset:** One tenth covtype dataset <br>
**Scoring:** mean score across 3 iterations of `StratifiedShuffleSplit` <br>
**n_estimators:** 50 <br>

MCAR

![](https://i.imgur.com/4c0AFUJ.png)

MNAR

![](https://i.imgur.com/e1WQh1x.png)

<hr>

##### Personal TODO

- [x] Introduce a [`drop_value`](https://gist.github.com/rvraghav93/75b80c76eadf6b7dfdcc) (link) method to generate missingness based on the preset levels of MCAR-ness(?)/MNAR-ness(?). Imporantly allow successive addition of missing samples with exact missing fractions.
- [x] Run on covtype comparing my implementation with imputation (and end up hating imputation.)
- [ ] See if we can heuristically set the missing values to take a random direction if the entire data(or more feasibly the current split) seems to have values missing completely at random. (Intuitively it sounds promising to me...)
- [x] Clean up the code and start committing it neatly
- [ ] Read [simonoff's paper](http://people.stern.nyu.edu/jsimonof/jmlr10.pdf) properly and summarize.
- [ ] Compare with [rpart](https://cran.r-project.org/web/packages/rpart/rpart.pdf) - How to send the missing data to/from python. How to combine sklearn's imputation and rpart in a pipeline?!
- [x] See if there is a good established way to measure MCAR-ness (ratio of missing fraction for the correlated labels is a decent way to measure MCAR-ness)

<hr>

##### References
1. [Simonoff's paper comparing different methods on **BINARY RESPONSE DATA**](http://people.stern.nyu.edu/jsimonof/jmlr10.pdf) > 100 citations
2. [Handling Missing Values when Applying Classification Models](http://www.jmlr.org/papers/volume8/saar-tsechansky07a/saar-tsechansky07a.pdf) > 100 citations
3. Ding Simonoff PHD thesis - should find the actual link or is it same as 1 - Google books [link](https://books.google.fr/books?id=XP3IgeZF2X4C&pg=PA33&lpg=PA33&dq=kim+yates+missing+method&source=bl&ots=dzbMSq0Lkb&sig=Mlr8nM09gsoAK9zwl8rwvjQsgQM&hl=en&sa=X&ved=0ahUKEwjo9tmqwpPLAhXIBBoKHQNuCJQQ6AEIKTAC#v=onepage&q=kim%20yates%20missing%20method&f=false)
4. [Missing Data Imputation for Tree based methods - 2006 - Yan He, UCLA](http://statistics.ucla.edu/system/resources/BAhbBlsHOgZmSSJOMjAxMi8wNS8xNC8xNV8zOV8zN181MDhfTWlzc2luZ19EYXRhX0ltcHV0YXRpb25fZm9yX1RyZWVfYmFzZWRfTW9kZWxzLnBkZgY6BkVU/Missing%2520Data%2520Imputation%2520for%2520Tree-based%2520Models.pdf)
5. http://sci2s.ugr.es/keel/pdf/specific/capitulo/IFCS04r.pdf - Compares various imputation methods and case deletion.

<hr>

###### Last Resort
- [x] Silently break the imputation and claim that my method is finally better than imputation (ofcourse kidding)
