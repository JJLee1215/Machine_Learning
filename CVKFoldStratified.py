'''
"I have finally built a function to compare between KFold and StratifiedKFold. :)" 

Purpose: Understanding cross validation, K-Fold and startify.

1. K-Fold Simulation.
1.1 K-Fold.
1.2 K-Fold with stratified.

2. Application of K-Fold: Comparison between Kfold w/ and w/o Stratified.

3. Cross validation.
4. Gridsearch: hyperparameter tunning.
'''

# 1. K-Fold simulation

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

arr = np.arange(1,21)

def test_ex(data, cv = 5, shuffle = False):
    folds = KFold(n_splits = cv, shuffle = shuffle) # How can I guess return value?
    for train_idx, valid_idx in folds.split(data):
        print(f'eval{data[valid_idx]} training{data[train_idx]}')

print("shuffle = False")
print(test_ex(arr, shuffle = False))
print("shuffle = True")
print(test_ex(arr, shuffle = True))


arr = np.array(['악성1', '악성2', '악성3', '악성4','악성5'] + ['양성']*45)

def test_st(data, cv = 5):
    y = data
    x = np.arange(len(y))
    folds = StratifiedKFold(n_splits = cv)
    for train_idx, valid_idx in folds.split(x, y):
        print(f'eval{y[valid_idx]} training{y[train_idx]}')
        
print("StratifiedKFold")
print(test_st(arr))

# 2. Application of K-Fold: Comparison between Kfold w/ and w/o Stratified.
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, y.shape)


def foldchoice(data, target, folds = KFold, n_splits = 5):
    X = data
    y = target

    if folds == KFold:
        folds = KFold(n_splits = n_splits)
    else:
        folds = StratifiedKFold(n_splits = n_splits)
            
    total_accs = []

    for train_idx, val_idx in folds.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        total_accs.append(acc)
    return print(f'Training CV Reuslt : {np.mean(total_accs)}')

foldchoice(X, y, KFold, n_splits = 5) # 0.906
foldchoice(X, y, StratifiedKFold, n_splits = 5) # 0.966


# 3. Cross validation.
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = 5, verbose = True)
print('Cross validation')
print(scores, np.mean(scores))

# 4. Hyperparameter Tunning by Gridsearch
# I omiited randomizedsearchCV

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 45, stratify = y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = DecisionTreeClassifier()

grid = GridSearchCV(
    model,
    {
        'max_depth' : [1, 2, 3],
        'min_samples_split' : [2, 3]
    },
    cv = 5,
    refit = True,
    return_train_score= True 
    )

grid.fit(X_train, y_train)

logs = pd.DataFrame(grid.cv_results_)
#print(logs)

print(logs[['params', 'mean_test_score', 'rank_test_score']])
print(grid.best_params_)

print(grid.score(X_test, y_test))

# val: 0.946 // test: 0.947 => the training method was suitable.
