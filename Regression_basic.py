'''
1. Types of regularization

1.1 L1 Regularization:
    - Utilizes the Lasso algorithm.
    - Applies a penalty to the absolute value of the regression coefficients.
    - Eliminates less effective regression coefficients by setting them to zero.
    - Reduces the number of features, preventing overfitting.
1.2 L2 Regularization:
    - Utilizes the Ridge algorithm.
    - Applies a penalty to the square of the regression coefficients.
    - Minimizes regression coefficients, preventing overfitting.

1.3 L1, L2 Regularization:
    - Utilizes the Elastic algorithm, combining features of both L1 and L2 regularization.
    - The perfomance is not good.

1.4 Alpha Cost Function:
    - Can be used as a parameter in the regularization techniques.
   
2. Algorithms.
2.1 Simple linear regression 
2.2 Ridge 
2.3 Lasso
2.4 Elasticnet
2.5 Logistc: the categorical data is used as dependent variables.
2.6 Ensemble

3. Loss function
3.1 MAE: Mean Absolute Error
3.2 MSE: Mean Squared Error
3.3 RMSE: Root Mean Squared Error
3.4 MSLE: Means Squared Log Error
3.5 RMSLE: Root Means Squared Log Error 
3.6 R^2: 
     
'''
# Test for loss funcitons
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

y_true = np.arange(16)
y_pred = np.arange(16)
np.random.shuffle(y_pred)

def show_all_reg_metric(true, pred):
    print('mae', mean_absolute_error(true, pred))
    
    print('mse', mean_squared_error(true, pred))
    print('rmse', np.sqrt(mean_squared_error(true, pred)))
    
    print('mlse', mean_squared_log_error(true, pred))
    print('rmlse', np.sqrt(mean_squared_log_error(true, pred)))
    
    print('r2', r2_score(true, pred))

show_all_reg_metric(y_true, y_pred)

