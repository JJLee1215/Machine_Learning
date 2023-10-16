import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from string import ascii_lowercase

# 1. 데이터 불러오기
with open('./res/freqs_labels.json') as f:
    data = json.load(f)
    
train_data, test_data = data

# 2. 데이터의 차원을 확인한다.
print(type(train_data))
print(train_data.keys())
print(len(train_data), len(train_data['freqs']), len(train_data['labels']))

# 3. Data frame으로 변환한다.
df_freqs = pd.DataFrame(train_data['freqs'], columns = list(ascii_lowercase))
print(df_freqs.shape)
print(df_freqs.head(2))

df_freqs['label'] = train_data['labels']
print(df_freqs.head(2))

# 4. EDA 데이터 시각화
df_train_pv = df_freqs.pivot_table(index = df_freqs.label)
print(df_train_pv)


plt.style.use('ggplot')
df_train_pv.T.plot(kind = 'bar', subplots = True, figsize = (20, 8), ylim = (0.0, 0.25))
plt.show()



df_train_pv.T.plot(kind = 'line', figsize = (20,8))
plt.show()



df_freqs.label.unique()
df_freqs[df_freqs.label == 'en']['a']

for ch in ascii_lowercase:
    print(ch)
    for na in df_freqs.label.unique():
        print(na)
        tmp = df_freqs[df_freqs.label == na][ch]
        tmp.plot(kind = 'hist', alpha = 0.5, label = na) 
        
    plt.legend()
    plt.suptitle(f'{ch}\' histogram')
    plt.show()
    break

