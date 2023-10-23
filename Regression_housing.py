'''
Regression Coefficient Estimation Methods
1. Ordinary Least Squares(OLS)
    - It minimizes the sum of squared differences between observed and predicted values.
    - Objective: the goal is to minimize the residual sum of squares(RSS) to find the best-fitting line through the data points.
2. Maximum Likelihood Estimation (MLE)
    - It finds the parameter values that maximize the likelihood function, 
        which measures the probability of observing the given data under the assumed statistical model.
    - Obejctive: MLE aims to find the parameter values that make the observed data most probable, 
        assuming a specific probabilistic model.
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

house = fetch_california_housing()
# print(house.DESCR) 
# print(house.keys()) dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])


# 1. DataFrame( data + target)
house_df = pd.DataFrame(house.data, columns = house.feature_names)
house_df[house.target_names[0]] = house.target
print(house_df.shape)

# 2. EDA - Correlation
fig, axs = plt.subplots(figsize = (16,8), ncols = 4, nrows = 2)
feats = house.feature_names

for idx, feat in enumerate(feats):
    sns.regplot(x = feat, y = house.target_names[0], data = house_df,
                ax = axs[int(idx/4)][idx%4])
    
plt.show()

# 3. Learning
# 3.1 Base line - simple linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 3.1.1 Train/Test
X = house_df.drop([house.target_names[0]], axis = 1)
y = house_df[house.target_names[0]]
# print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3.1.2 Estimation of the regression coefficients
lr = LinearRegression() # OLS basis
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# print(y_pred)

# 3.1.3 Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
score = lr.score(X_test, y_test)
 # r2_score == lr.score
print("mse : {}, rmse : {}, r2 : {}, score : {}".format(mse, rmse, r2, score))

# 3.2 Cross validation
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
neg_mse = cross_val_score(lr, X_train, y_train, scoring = 'neg_mean_squared_error', cv =5)

# 4. Regularization
# 4.1 Ridge model
from sklearn.linear_model import Ridge

alphas = [0, 1, 10, 100]

for alpha in alphas:
    ridge = Ridge(alpha = alpha)
    neg_mse = cross_val_score(ridge, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 5)
    rmse = np.sqrt(-1*neg_mse)
    print(alpha," : " ,np.mean(rmse))
# There is no improvement in this model.
