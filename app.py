from flask import Flask, request, jsonify
import requests
import numpy as np
import keras
import cv2
import joblib
import urllib.request
import ssl
from PIL import Image
from matplotlib import pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

#Flask 객체 인스턴스 생성
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/search/image', methods=['POST'])
def search():
    lists = request.args['file_name']
    lists = lists.split(',')
    
    # model load
    model = keras.models.load_model('model_8.h5')
    le = joblib.load('label_encoder_8.pkl')
    
    print(lists[0])
    print(lists[1])
    
    urllib.request.urlretrieve(lists[0], '/Users/jooheekim/Desktop/school/capstone/ML_server/front.jpg')
    urllib.request.urlretrieve(lists[1], '/Users/jooheekim/Desktop/school/capstone/ML_server/back.jpg')

    front_image = cv2.imread('front.jpg')
    front_image_rgb = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
    front_image_resized = cv2.resize(front_image_rgb, (128, 128))

    back_image = cv2.imread('back.jpg')
    back_image_rgb = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
    back_image_resized = cv2.resize(back_image_rgb, (128, 128))

    # 두 이미지를 합칩니다.
    merged_image = np.hstack((front_image_resized, back_image_resized))
    merged_image = merged_image / 255.0  # 정규화
    merged_image = np.expand_dims(merged_image, axis=0)  # 모델 예측을 위한 차원 확장

    # 예측을 수행합니다.
    prediction = model.predict(merged_image)
    predicted_class = le.inverse_transform([np.argmax(prediction)])

    print(f"Predicted class is: {predicted_class[0]}")

    predicted_item = predicted_class[0].tolist()
    
    data = ['스피자임에스정']
    # data = { "name" : '가스디알정50밀리그램(디메크로틴산마그네슘)' }
    # for list in lists:
    #     data.append(list)
    

    return jsonify({
        'result': predicted_item
        # 'result': data
    })
    

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()