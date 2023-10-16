import joblib
import os
from string import ascii_lowercase
import re

clf = joblib.load(os.path.join(os.path.dirname(__file__),
                               'lang_predict.ml'))

label = joblib.load(os.path.join(os.path.dirname(__file__),
                               'lang_predict.lb'))

def encode_freqs_data(src:str) -> list:
    '''
    - 텍스트 원문 -> 빈도계산 및 정답 추출 -> 특정 구조로 리턴.
    - parameters
        - src : 번역 요청 원문 텍스트
    - returns
        - list : [[0.000, ...]]
    '''
    print(src)
    
    ALPHA_LEN = len(ascii_lowercase)
    STD_ASCII_A = ord('a')
    
    text = src.lower()
    p = re.compile('[^a-zA-Z]*')
    text = p.sub('', text)
    counts = [0] * ALPHA_LEN
    for ch in text:
        counts[ord(ch) - STD_ASCII_A] += 1
    total_count = len(text)
    counts_norms = list(map(lambda x: x/total_count, counts))
    return [counts_norms]

def predict( data:list ) -> dict:
    pred_y = clf.predict( data )
    print(pred_y[0])
    
    return {
        "success" : 1,
        "code" : pred_y[0],
        "ko" : label[pred_y[0]]
    }

if __name__ == '__main__':
    # 단위 테스트 용도
    sample_data = '''
    The Private Case is a collection of erotica and pornography held initially by the British Museum and, from 1973, by the British Library. 
    The collection began between 1836 and 1870 and grew from the receipt of books from legal deposit, 
    and from requests made to the police following seizures of obscene material. 
    Access to the material in the Private Case was restricted. 
    At its height numbering some 4,000 items, the contents of the Private Case shrank as works were moved to the general collection, 
    and grew with the arrival of bequests and donations from collectors. 
    From 1964, reflecting the changing social mores of the time, 
    the library began to review the Private Case, allowing public access to its contents, a process that was completed in 1983. 
    There have been no new entries since 1990 and all new erotic and pornographic material is put on open access in the general collection. 
    There is no restriction on access to Private Case material, except for some items which are in a fragile condition.  '''
    
    preprocessing_data = encode_freqs_data(sample_data)
    result = predict(preprocessing_data)
    print(result)
    
