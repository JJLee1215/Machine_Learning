# 목표: res 폴더에 있는 데이터 정제
# 1. 정답 추출 함수를 만든다.
# 2. 피처 데이터 추출 함수를 만든다.
# 3. 최종 형태를 만들어 train/test로 저장한다.
# 4. 데이터 파일을 덤프한다. : 일반적으로 pickle 혹은 json 파일로 저장.

import glob

files = glob.glob('./res/train/*.txt')
files.sort()
#print(files)

# 1. 정답 추출
# 1.1 정답을 어떻게 추출할지 확인한다.
print(files[0].split('\\')[-1][:2])
# 1.2 정답을 추출하는 함수를 구성한다.
def get_label(filename : str) -> str:
    return filename.split('\\')[-1][:2]

#print(get_label(files[0]))

# 2. 피처 데이터 추출
import re
from string import ascii_lowercase

with open(files[0], 'r', encoding = 'utf-8') as f:
    text = f.read().lower()
    p = re.compile('[^a-zA-Z]*')
    text = p.sub('', text)
    print(text[:100])
      
    ALPHA_LEN = len(ascii_lowercase) # 변수를 대문자 표시: 이 값은 고정값 이라는 암묵적인 룰
    counts = [0] * ALPHA_LEN
    STD_ASCII_A = ord('a')
    
    for ch in text:
        counts[ord(ch) - STD_ASCII_A] += 1
    
    total_count = len(text)
    counts_norms = list(map(lambda x: x/total_count, counts))

print(sum(counts_norms), counts_norms)
        
# 3. 최종 형태를 만들어 train, test 형태로 저장할 수 있는 함수 만들기.
import glob
import re
from string import ascii_lowercase

def encode_freqs_data(dir:str = 'train') -> dict: 
    # 이렇게 표현 해주는게 매우 중요함.! 어차피 나중에 문서화 해야함.
    # 아래처럼 주석을 달아주면 나의 함수에 대해 설명할 수 있다. 
    '''
    - 텍스트 원문 -> 빈도계산 및 정답 추출 -> 특정 구조로 리턴 
    - parameters
        - dir : 용도 [train | test]
    - returns
        - dict : {'freqs' : [], 'labels' : []}
    '''
    freqs = list()
    labels = list()
    
    dir_path = f'./res/{dir}/*.txt'
    files = glob.glob(dir_path)
    
    ALPAH_LEN = len(ascii_lowercase)
    STD_ASCII_A = ord('a')
    
    for file in files:
        label = get_label(file)
        labels.append(label)
        
        with open(file, 'r', encoding = 'utf-8') as f:
            text = f.read().lower()
            p = re.compile('[^a-zA-Z]*')
            text = p.sub('', text)
            counts = [0] * ALPHA_LEN
            for ch in text:
                counts[ord(ch) - STD_ASCII_A] += 1
            total_count = len(text)
            counts_norms = list(map(lambda x: x/total_count, counts))
        freqs.append(counts_norms)
    
    return {
        'freqs' : freqs,
        'labels' : labels
    }

train_raw_data = encode_freqs_data()
test_raw_data = encode_freqs_data('test')
print(len(train_raw_data['freqs']),len(train_raw_data['labels']))

# 4. 데이터 파일을 덤프한다.
import json

with open('./res/freqs_labels.json', 'w') as f:
    json.dump([train_raw_data, test_raw_data], f)
    