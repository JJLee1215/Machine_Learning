# 목표: 위키피디아에서 텍스트를 불러와 정제

# 1. 타겟 사이트 지정
# 2. url을 통해 사이트에 접속
# 3. Beautiful soap을 이용해 html을 parsing 해준다.
# 4. 내가 원하는 부분을 선택한다 (크롬 -> 개발자도구)
#   - 1) ctr + shift + C 를 통해 원하는 부분의 태그를 찾는다.
#   - 2) id는 유일하므로 id를 기준으로 해당 부분을 검색하면 좋다.
#   - 3) 여기서는 <p> 태그들로 문장이 이루어져 있으므로 <p> 태그의 갯수로 모든 문장을 넣었는지 확인 할 수 있다.
#   - 4) 'div#mw-content-text p'
# 5. 모든 text자료를 리스트에 담은 후, 하나의 말뭉치로 만들어 준다.
# 6. 정규표현 식 등을 활용하여 불순물을 제거한다.

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

target_site = "https://en.wikipedia.org/wiki/Tesla_Model_Y"
res = urlopen(target_site)
soup = BeautifulSoup(res, 'html5lib')

'''
# 이렇게 해서 잘 불러왔는지 한번 테스트 해본다.
for p in soup.select('div#mw-content-text p')[:2]:
    print(p.text.strip())
'''

texts = [p.text.strip() for p in soup.select('div#mw-content-text p')]
#print(len(texts)) #p 태그의 갯수와 일치하는지 확인하여 누락된 데이터가 있는지 검토한다.
#print(texts)

data = ''.join(texts)
#print(len(data))

p = re.compile('[^a-zA-Z]*')
preprocessed_data = p.sub('', data)
print(preprocessed_data)