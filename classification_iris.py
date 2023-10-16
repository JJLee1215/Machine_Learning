from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

raw_data = load_iris()
#print(raw_data)
#print(raw_data.keys())
'''
1. raw_data는 dictionary type으로 정보가 저장되어있음.
   dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
2. data = X, target = y, target_names =  array(['setosa', 'versicolor', 'virginica']
3. X = (150,4)
4. y = (150,)
'''


# 1. 판다스로 읽어오기
df = pd.DataFrame(data = raw_data.data, columns = raw_data.feature_names)
df['target'] = raw_data.target

#print(df)

# 2. X와 y를 분리하기
X = df.iloc[:,:-1]
y = df.iloc[:,-1] # <- 차원을 축소 시키기 위해
#print(X.shape, y.shape)

# 3. 목표 y == 1 'versicolor'인지 아닌지를 구분하는 binary classification
from sklearn.model_selection import train_test_split
# 3.1 Train, test 나누기
y_ = y == 1
#print(y_.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.25, random_state = 100)

# 3.2 True, False의 비율이 잘 맞게 나뉘었는지 확인.
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(X_train, y_train)
#print(np.unique(y_train.values, return_counts = True), np.unique(y_test.values, return_counts = True))

# 3.3 알고리즘 선정 

# 로지스틱 회귀, 이진분류용, 베이스라인 모델용도로 사용
from sklearn.linear_model import LogisticRegression
# 결정 트리, 다중 분류용, 앙상블 기본 베이스
from sklearn.tree import DecisionTreeClassifier
# 랜덤포레스트 (동일알고리즘 n개 사용)
from sklearn.ensemble import RandomForestClassifier
# 데이터가 텍스트(자연어)일때 주로 사용한다.
from sklearn.naive_bayes import GaussianNB
# 서포트 벡터 머신, 이진 분류용
from sklearn.svm import SVC

als = {
    'LogisticRegression':(LogisticRegression(), '-'),
    'DecisionTreeClassifier':(DecisionTreeClassifier(max_depth = 5), '--'),
    'RandomForestClassifier':(RandomForestClassifier(max_depth = 5, max_features = 1, n_estimators = 10), '.-'),
    'GaussianNB':(GaussianNB(),':'),
    'SVC':(SVC(probability = True), '-' ),
}

# 3.4 알고리즘 별 학습 수행.
from sklearn.metrics import roc_curve, auc

# 차트 모양
plt.figure(figsize = (5,5))

#LogisticRegression().predict_proba()

for key, (model, line_style) in als.items():

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)
    
    model.predict_proba()

    pred_t = pred[ : , -1 ] # <-- 왜 이렇게 되는지 확인할 것
    # print(pred_t)
    # 성능 평가
    fpr, tpr, _ =roc_curve(y_test.values, pred_t,)

    # 차트 그리기
    plt.plot(fpr, tpr, line_style, label = key)
    # auc값 출력

    print(key, auc(fpr, tpr))
    #break

# 플로팅
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 차트 출력
plt.show()
