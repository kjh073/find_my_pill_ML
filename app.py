from flask import Flask, request, jsonify
import requests
import numpy as np
import json
import keras
import cv2
# from sklearn.preprocessing import LabelEncoder
import joblib
from keras.utils import to_categorical
import urllib.request
import ssl

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
    
    # front_image = requests.get(lists[0]).content
    # back_image = requests.get(lists[1]).content
    #numpy형식으로 변환
    # image_array_f = np.frombuffer(front_image, np.uint8)
    # image_array_b = np.frombuffer(back_image, np.uint8)
    # image_f = cv2.imdecode(image_array_f, cv2.IMREAD_COLOR)
    # image_b = cv2.imdecode(image_array_b, cv2.IMREAD_COLOR)
    # print("front", front_image.text, flush=True)
    
    # model load
    shape_model = keras.models.load_model('my_model3.h5')
    item_model = keras.models.load_model('my_model4.h5')
    shape_le = joblib.load('labelencoder.pkl')
    item_le = joblib.load('item_label_encoder.pkl')
    
    urllib.request.urlretrieve(lists[0], '/Users/jooheekim/Desktop/school/capstone/ML_server/front.jpg')
    urllib.request.urlretrieve(lists[1], '/Users/jooheekim/Desktop/school/capstone/ML_server/back.jpg')
    
    back_img = cv2.imread('back.jpg')
    front_img = cv2.imread('front.jpg')

    back_img = cv2.resize(back_img, (128, 128))
    front_img = cv2.resize(front_img, (128, 128))

    back_img = np.expand_dims(back_img, axis=0) / 255.  # 이미지를 정규화합니다.
    front_img = np.expand_dims(front_img, axis=0) / 255.  # 이미지를 정규화합니다.
    
    # 첫 번째 모델을 사용하여 뒷면 이미지의 알약 형태를 예측합니다.
    shape_pred = shape_model.predict(back_img)
    shape_pred_argmax = np.argmax(shape_pred, axis=-1)

    # 예측된 알약 형태를 One-hot encoding으로 변환합니다.
    shape_pred_onehot = to_categorical(shape_pred_argmax, num_classes=len(item_le.classes_))
    
   # 두 번째 모델을 사용하여 앞면 이미지와 알약 형태 예측 결과를 바탕으로 품목을 예측합니다.
    item_pred = item_model.predict([front_img, shape_pred_onehot])

    # 가장 확률이 높은 클래스의 인덱스를 선택합니다.
    item_pred_argmax = np.argmax(item_pred, axis=-1)

    # 예측된 알약 형태를 One-hot encoding으로 변환합니다.
    shape_pred_onehot = to_categorical(shape_pred_argmax, num_classes=len(shape_le))

    # 예측된 형태와 품목의 실제 이름을 가져옵니다.
    predicted_shape = list(shape_le.keys())[shape_pred_argmax[0]]
    predicted_item = item_le.classes_[item_pred_argmax[0]]

    print("Predicted shape:", predicted_shape)
    print("Predicted item:", predicted_item)

    predicted_item = predicted_item.tolist()
    
    # data = ['스피자임에스정']
    # data = { "name" : '가스디알정50밀리그램(디메크로틴산마그네슘)' }
    # for list in lists:
    #     data.append(list)
    

    return jsonify({
        'result': predicted_item
    })
    

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()