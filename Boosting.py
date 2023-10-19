'''
Boosting
1. process
- Sequantially train and predict using multiple classifiers.
- Check the prediction errors.
- Improve accuracy by assigning weights to error data.


2. Types
: AdaBoost, Gradient Boost Machine, XGBoost, 

'''

# 1. Loading data set
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    stratify = cancer.target,
                                                    random_state = 0)

# 2. Boosting
# 2.1 Gradient Boost Machine (GBM)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state = 0)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

# 2.2 XGBoost
import xgboost as xgb

#print(xgb.__version__)

data_train = xgb.DMatrix(data = X_train, label = y_train)
data_test = xgb.DMatrix(data = X_test, label = y_test)

params = {
    'objective' : 'binary:logistic',
    'eval_metric' : 'logloss',
    'max_depth' : 6,
    'eta' : 0.1,
    'early_stopping_rounds' : 10,
}

data_val = data_test

eval = data_test # it will be replaced evaluation data later
evals = [(data_train, 'train'),(data_val, 'eval')]

xgb_model = xgb.train(
    params = params,
    dtrain = data_train,
    num_boost_round = 400,
    evals = evals
)

y_pred = xgb_model.predict(data_test)

y_preds = [1 if x > 0.5 else 0 for x in y_pred]

print(accuracy_score(y_test, y_preds))

from xgboost import plot_importance
import matplotlib.pyplot as plt

_, ax = plt.subplots(figsize = (14,10))
plot_importance(xgb_model, ax)
print(plt.show())

## Cross-validation with XGBoost
xgb.cv(params = params, dtrain = data_train, num_boost_round = 400, nfold =3)
