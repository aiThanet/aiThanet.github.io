---
layout: post
title: XGBoost Parameter Tuning Note
date: 2019-07-02
Author: aiThanet
categories:
tags: [decision tree, data science, note]
comments: true
---

source : <https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/>

### What is XGBoost

> XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions.
> XGBoost is an implementation of over gradient boosted decision trees which designed to improve speed and performance.

### XGBoost Advantage

1. Regularization : Standard GBM has no regularization to reduce overfitting problem.
2. Parallel Processing : much easy to scale, more detail on how [parallel](http://zhanpengfang.github.io/418home.html).
3. High Flexibility : allow to custom optimization objectives and evaluation criteria.
4. Handling Missing Values : handling missing value built-in.
5. Tree Pruning
6. Built-in Cross-Validation
7. Continue on Existing Model

#### XGBoost Parameters

##### General Parameters: Guide the overall functioning

1. booster [default=gbtree]

   - It has 2 options: gbtree: tree-based models or gblinear: linear models

2. silent [default=0]:

   - Silent mode is activated is set to 1, i.e. no running messages will be printed.
   - It’s generally good to keep it 0 as the messages might help in understanding the model.

3. nthread [default to maximum number of threads available if not set]

##### Booster Parameters: Guide the individual booster (tree/regression) at each step

Consider only tree booster here because it always outperforms the linear booster and thus the later is rarely used.

1. eta [default=0.3]

   - Analogous to learning rate in GBM
   - Makes the model more robust by shrinking the weights on each step
   - Typical final values to be used: 0.01-0.2

2. min_child_weight [default=1]

   - Defines the minimum sum of weights of all observations required in a child.
   - This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
   - Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
   - Too high values can lead to under-fitting hence, it should be tuned using CV.

3. max_depth [default=6]

   - The maximum depth of a tree, same as GBM.
   - Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample
   - Should be tuned using CV
   - Typical values: 3-10

4. max_leaf_nodes

   - The maximum number of terminal nodes or leaves in a tree.
   - Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
   - If this is defined, GBM will ignore max_depth.

5. gamma [default=0]

   - A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
   - Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.

6. max_delta_step [default=0]

   - In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
   - Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
   - This is generally not used but you can explore further if you wish.

7. subsample [default=1]

   - Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
   - Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
   - Typical values: 0.5-1

8. colsample_bytree [default=1]

   - Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
   - Typical values: 0.5-1

9. colsample_bylevel [default=1]

   - Denotes the subsample ratio of columns for each split, in each level.
   - I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.

10. lambda [default=1]

    - L2 regularization term on weights (analogous to Ridge regression)
    - This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.

11. alpha [default=0]

    - L1 regularization term on weight (analogous to Lasso regression)
    - Can be used in case of very high dimensionality so that the algorithm runs faster when implemented

12. scale_pos_weight [default=1]
    - A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.

##### Learning Task Parameters

1. objective [default=reg:linear]

   - This defines the loss function to be minimized. Mostly used values are:
     - binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
     - multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
       - you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
     - multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.

2. eval_metric [ default according to objective ]

   - The metric to be used for validation data.
   - The default values are rmse for regression and error for classification.
   - Typical values are:
     - rmse – root mean square error
     - mae – mean absolute error
     - logloss – negative log-likelihood
     - error – Binary classification error rate (0.5 threshold)
     - merror – Multiclass classification error rate
     - mlogloss – Multiclass logloss
     - auc: Area under the curve

3. seed [default=0]

   - The random number seed.
   - Can be used for generating reproducible results and also for parameter tuning.
