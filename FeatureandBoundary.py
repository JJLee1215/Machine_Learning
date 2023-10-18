'''
Identifying important features and reducing less important ones are crucial steps 
in dimensionality reduction of the feature vector. 

Additionally, I attempted to display the classification boundaries on the graph
'''


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 1. Simple method

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# 2. Feature Engineerig
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print(model.feature_importances_) 
print(iris.feature_names)

sns.barplot(x = model.feature_importances_, y = iris.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.show()

# 3. Classification Boundary
def show_clf_area(clf, X, y, seg_num = 50):
    '''
    - clf: model
    - X : feature data
    - y : target data
    '''
    _, ax = plt.subplots() # this returns fig and axis. But I choosed only axis
    ax.scatter(X[:,2], X[:,3] , c = y, zorder = 5, s = 25, edgecolors = 'k', cmap = 'rainbow')
    
    x_limit_s, x_limit_e = ax.get_xlim()
    y_limit_s, y_limit_e = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(x_limit_s, x_limit_e, num = seg_num),
                          np.linspace(y_limit_s, y_limit_e, num = seg_num),)
    
    # xy = np.concatenate[xx.ravel(), yy.ravel()]
    xy = np.c_[xx.ravel(), yy.ravel()]
    
    clf.fit(X[:,2:], y) # training with only two important features
    
    y_pred =clf.predict(xy)
    
    ax.scatter(xy[:, 0], xy[:, 1], zorder = 1, s = 1, c = y_pred)
    
    plt.show()
         
show_clf_area(DecisionTreeClassifier(random_state = 0), X_train, y_train) 
    