'''
*** pipeline 사용법 다시 익히기 ***

Purpose: hyper parameter tunning and pipeline

1. Check the data dimensions and characters
1.1 Data loading
1.2 Check data format through keys (almost dictionary form)
1.3 Check dimension X and y
 
2. Data preprocessing
2.1 scaling: Scaling ensures equal contribution from each feature.

3. Model 

4. Prediction and Evaluation

5. Build a piple line to solve an issue

''' 

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. Check the data dimensions and characters
cancer = load_breast_cancer()
#print(cancer.DESCR)
#print(type(cancer))
#print(cancer.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 
    # 'feature_names', 'filename', 'data_module'])

X = cancer.data
y = cancer.target

print(f'X.shape: {X.shape}, y.shape: {y.shape}')
    # Dataset = {(569,31)}, X in R^30, y in R^1, Dataset size#: 569

# 1.2 Degree of data bias.
print(np.unique(y, return_counts = True))
    # (array([0, 1]), array([212, 357], dtype=int64))
    # -> mal: 37%, benign: 63%
    # -> when training data, you should keep the ratio by stratify

# 2. Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, stratify = y)
    # Keep ratio by stratify = y
print(f'X_train : {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
print(f'y_train: {np.unique(y_train, return_counts = True)}, y_test: {np.unique(y_test, return_counts = True)}')

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train_scaled: {}'.format(X_train_scaled))
    
# 2. model
from sklearn.svm import SVC

clf = SVC()
clf.fit(X_train_scaled, y_train)

# 3. Prediction and Evaluation 
from sklearn.metrics import accuracy_score, recall_score

X_test_scaled = scaler.transform(X_test)
y_pred = clf.predict(X_test_scaled)

print("score: {}".format(clf.score(X_test_scaled, y_test)))
print(f'accuracy: {accuracy_score(y_test, y_pred)}, recall: {recall_score(y_test, y_pred)}')

      
################# 

# 4. Hyper parameter tunnning <- GridSearch, RandomizedSearchCV, ...
# 4.1 GridSearch
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

grid = GridSearchCV(SVC(), param_grid, cv = 5)
grid.fit(X_train_scaled, y_train)

print("best_estimator_: {}, best_score_: {}, best_params_: {}".format(
    grid.best_estimator_, grid.best_score_, grid.best_params_))
print("best_estimator_.score: {}".format(grid.best_estimator_.score(X_test, y_test)))

#print(grid.cv_results_.keys())
# I think it is better to know what types of data are in grid.cv
# dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_C', 
# 'param_gamma', 'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 
# 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'])

'''
** Issue ** 
When separating the train set into train and evaluation sets, 
the evaluation set is already included in the initial scaling. 
As a result, the effectiveness of the evaluation is reduced.

'''
# 5. Build a pipeline

from sklearn.pipeline import make_pipeline, Pipeline

# steps = [('name', objects)]
pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', SVC())])
print("pipe:{}".format(pipe))
# Configuration: cross validation + hyper parameter tunning + n algorithms + n preprocessing
# Finding a best option
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# n alogrithms -> [{}1, {}2, ..., {}n]
param_grid = [
    {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'clf': [SVC()],
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'clf__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    {
    'scaler': [None],
    'clf': [RandomForestClassifier(n_estimators=90)],
    'clf__max_features': [1, 2, 3]
    }
]

print("param_grid:{}".format(param_grid))

# configuraiton of gridsearch
grid = GridSearchCV(pipe, param_grid, cv = 5)

grid.fit(X_train, y_train) # <-- you do 'not' adjust scaling.

print(f"grid.best_estimator_: {grid.best_estimator_}, "
      f"grid.best_score_: {grid.best_score_}, "
      f"grid.best_params_: {grid.best_params_}")

print(f"grid.best_estimator_.score: {grid.best_estimator_.score(X_test, y_test)}")