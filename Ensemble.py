'''
I studied the voting and bagging methods. 

Additionally, I implemented the display of feature importances.
'''

# 1. Voting

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                   cancer.target,
                                                   stratify = cancer.target,
                                                   random_state = 0)

clf_lr = LogisticRegression()
clf_knn = KNeighborsClassifier()
clf_dtc = DecisionTreeClassifier()

vc_clf = VotingClassifier([('LR',clf_lr), ('KNN',clf_knn), ('DC',clf_dtc)],
                          voting = 'soft')

vc_clf.fit(X_train, y_train)

print(vc_clf.score(X_test, y_test))


# 2. Bagging

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 0)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))


# 3. Visualization of a feature importance 
import seaborn as sns
import matplotlib.pylab as plt

s_data = pd.Series(clf.feature_importances_, index = cancer.feature_names)

s_data.sort_values(ascending = False, inplace = True)
print(s_data)

sns.barplot(x = s_data, y = s_data.index)
plt.show()