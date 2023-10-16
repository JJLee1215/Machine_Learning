# html을 읽어서 랜더링 후 (필요시 데이터를 넣어 독적 구성) 응답 -> SSR(server side rendering)
from flask import Flask, render_template, request, jsonify

# 머신러닝 모델을 이용한 예측  처리 함수 가져오기.
from model import encode_freqs_data, predict
app = Flask(__name__)

@app.route('/')
def home():
    # render_template => templates 폴더 밑에서 index.html을 찾아서 읽음.
    # 데이터를 버무려서, 동적으로 구성해서 리턴
    return render_template('index.html')

@app.route('/detect_lang', methods = ['POST'])
def detect_lang():
    # 1. 클라이언트가 보낸 내용을 받는다. (post로 보냄) => 요청을 타고 들어온다!
    # request.form[] 보다는 request.form.get() 방식이 더 안정적이다.
    ori_text = request.form.get('ori_text')
    # 2. 데이터를 전처리 <- 머신러닝 초입에서 구현
    # 3. 모델 로드(서빙을 받음) <- 머신러닝
    # 4. 예측 수행 <- 머신 러닝
    # 5. 응답 데이터 구성
    return f'언어 감지 페이지 : {ori_text}'
@app.route('/ssgo', methods = ['GET', 'POST'])
def ssgo():
    if request.method == 'GET':
        return render_template('index2.html')
    else:
        # 1. 요청 데이터 가져오기
            # POST 처리
            # 전송한 데이터 추출
            # request.args.get('키') => GET 방식
        ori_text = request.form.get('ori_text')
        
        preprocessing_data = encode_freqs_data(ori_text)
        
        result = predict(preprocessing_data)
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True)