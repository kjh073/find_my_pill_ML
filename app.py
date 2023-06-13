from flask import Flask, request, jsonify
import requests
import numpy as np
import json
import keras
import cv2
from sklearn.preprocessing import LabelEncoder

#Flask 객체 인스턴스 생성
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/search/image', methods=['POST'])
def search():
    lists = request.args['file_name']
    lists = lists.split(',')
    
    front_image = requests.get(lists[0]).content
    back_image = requests.get(lists[1]).content
    #numpy형식으로 변환
    image_array_f = np.frombuffer(front_image, np.uint8)
    image_array_b = np.frombuffer(back_image, np.uint8)
    image_f = cv2.imdecode(image_array_f, cv2.IMREAD_COLOR)
    image_b = cv2.imdecode(image_array_b, cv2.IMREAD_COLOR)
    # print("front", front_image.text, flush=True)
    
    #model load, input, output
    model = keras.models.load_model('my_model2.h5')
    le = LabelEncoder()
    # le.fit()
    front_image_resized = cv2.resize(image_f, (128, 128))
    back_image_resized = cv2.resize(image_b, (128, 128))
    combined_image_resized = cv2.resize(np.hstack((front_image_resized, back_image_resized)), (256, 128))
    combined_image_input = np.expand_dims(combined_image_resized, axis=0)
    output = model.predict(combined_image_input)
    print("Output of model 2:", output)
    class_index2 = np.argmax(output)
    class_name2 = le.inverse_transform([class_index2])
    print("Predicted class of model 2:", class_name2)
    
    
    # data = ['스피자임에스정']
    # data = { "name" : '가스디알정50밀리그램(디메크로틴산마그네슘)' }
    # for list in lists:
    #     data.append(list)
    

    return jsonify({
        'result': output
    })
    

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()