##### Notes

* The PR - [scikit-learn/scikit-learn #5974](https://github.com/scikit-learn/scikit-learn/pull/5974)
* The missing values (in each split) are handled by deciding the best way to send them based on the impurtiy decrease.
* XGBoost handles the missing values in a very similar way. (https://github.com/dmlc/xgboost/issues/21#issuecomment-51982962) (Thanks to Jacob!)
* This is working for `RandomForestClassifier`, `DecisionTreeClassifier`, `BestSplitter` (dense only) and `ClassificationCriterion`.
* Yet to implement the this for `BestFirstTreeBuilder`, `RandomSplitter`, and `{Best|Random}SparseSplitter`
* Yet to implement the this for `RandomForestRegressor`, `DecisionTreeRegressor` and `RegressionCriterion`
* I wrote a `drop_value` function to successively introduce missing data. The code can be found [here](https://github.com/rvraghav93/miss_val_bench/blob/master/value_dropper.py).


<hr>

##### Some recent benchmarks

The benchmark notebook - https://github.com/rvraghav93/miss_val_bench/blob/master/missing_val_bench.ipynb

**Dataset:** 1/20 of the covtype dataset <br>
**Scoring:** mean score across 3 iterations of `StratifiedShuffleSplit` <br>
**n_estimators:** 50 <br>
**Comparing RF w/MV, Imp + RF w/o MV, XGBoost's RF w/MV, Imp + XGBoost's RF w/o MV**

1. When all the classes are present, and missing values across all the features correspond to one of the classes. (MNAR) (class 1)


  We can see a significant advantage with our method compared to the imputation. Also this implementation vs imputation is similar to XGBoost's RF vs imputation.
  
  The missingness, in this case, adds additional information (tells us which samples are label 1) and hence the performance increases with increasing missing fraction.
  
    ![MNAR](https://i.imgur.com/72l1FG8.png)

2. When the missing values are completely at random (MCAR).

  Our method performs very similar to imputation, while handling the missing data natively.
  
  As the missingness, in this case, is basically noise, you can see the performance drop with increasing missing fraction.
  
    ![MCAR](https://i.imgur.com/WaSZRwB.png)

<hr>

##### References

1. [Simonoff's paper comparing different methods on **BINARY RESPONSE DATA**](http://people.stern.nyu.edu/jsimonof/jmlr10.pdf) > 100 citations
2. [Handling Missing Values when Applying Classification Models](http://www.jmlr.org/papers/volume8/saar-tsechansky07a/saar-tsechansky07a.pdf) > 100 citations
3. Ding Simonoff PHD thesis - should find the actual link or is it same as 1 - Google books [link](https://books.google.fr/books?id=XP3IgeZF2X4C&pg=PA33&lpg=PA33&dq=kim+yates+missing+method&source=bl&ots=dzbMSq0Lkb&sig=Mlr8nM09gsoAK9zwl8rwvjQsgQM&hl=en&sa=X&ved=0ahUKEwjo9tmqwpPLAhXIBBoKHQNuCJQQ6AEIKTAC#v=onepage&q=kim%20yates%20missing%20method&f=false)
4. [Missing Data Imputation for Tree based methods - 2006 - Yan He, UCLA](http://statistics.ucla.edu/system/resources/BAhbBlsHOgZmSSJOMjAxMi8wNS8xNC8xNV8zOV8zN181MDhfTWlzc2luZ19EYXRhX0ltcHV0YXRpb25fZm9yX1RyZWVfYmFzZWRfTW9kZWxzLnBkZgY6BkVU/Missing%2520Data%2520Imputation%2520for%2520Tree-based%2520Models.pdf)
5. http://sci2s.ugr.es/keel/pdf/specific/capitulo/IFCS04r.pdf - Compares various imputation methods and case deletion.
