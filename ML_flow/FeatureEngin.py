# 1. 데이터 불러오기
# 2. 알고리즘 선택 및 학습
# 3. 예측
# 4. 성능 평가
# 5. 모델 덤프

# 1. 데이터 불러오기
import json

with open('./res/freqs_labels.json') as f:
    train_data, test_data = json.load(f)

# 훈련용 데이터: 2차원, 정답데이터: 1차원    
X = train_data['freqs']
y = train_data['labels']
print(len(X), len(X[0]), len(y))

# 2. 알고리즘 선택 및 학습
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X,y)

# 3. 예측
X_ = test_data['freqs']
y_ = test_data['labels']
print("예측: ", len(X_), len(X_[0]), len(y_))

pred_y = clf.predict(X_)
print(pred_y)

# 4. 성능 평가
from sklearn.metrics import accuracy_score

accuracy_score(y_, pred_y)

from sklearn.metrics import classification_report

print(classification_report(y_, pred_y))

# 5. 모델 덤프
import joblib

print("joblib: ", joblib.__version__)

# 모델 덤프
joblib.dump(clf, "./res/lang_predict.ml")

# 레이블 덤프
target = {'en': "영어", 'fr': "프랑스어", 'id': "인도네시아어", 'tl': "타칼리아어"}
joblib.dump(clf, "./res/lang_predict.lb")

print("dump completed")

