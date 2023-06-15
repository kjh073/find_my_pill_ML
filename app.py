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
    model = keras.models.load_model('model_9.h5')
    le = joblib.load('label_encoder_9.pkl')
    
    print(lists[0])
    print(lists[1])
    
    urllib.request.urlretrieve(lists[0], '/Users/jooheekim/Desktop/school/capstone/ML_server/front.jpg')
    urllib.request.urlretrieve(lists[1], '/Users/jooheekim/Desktop/school/capstone/ML_server/back.jpg')

    # 이미지를 불러오기
    img = Image.open('front.jpg')
    
    # 이미지 크기 구하기
    width, height = img.size

    # 자를 영역의 크기를 정하기
    crop_size = (1000, 1000)

    # 이미지 가운데를 기준으로 자를 영역을 정하기
    left = (width - crop_size[0])/2
    top = (height - crop_size[1])/2
    right = (width + crop_size[0])/2
    bottom = (height + crop_size[1])/2

    # 이미지를 자르기
    img_cropped = img.crop((left, top, right, bottom))

    # 자른 이미지를 저장하기
    img_cropped.save('cropped_image.jpg')

    # 저장된 이미지를 불러오기
    img = cv2.imread('cropped_image.jpg')
    original_img = img.copy()

    # 이미지의 크기를 가져오기
    height, width = img.shape[:2]

    # Canny edge detection 적용하기
    edges = cv2.Canny(img, threshold1=30, threshold2=100)

    # 가장 밝은 부분만 검출하기 위한 마스크 생성하기
    ret, mask = cv2.threshold(edges, 254, 255, cv2.THRESH_BINARY)

    # mask에서 알약의 왼쪽, 오른쪽, 위, 아래 부분 찾기
    ys, xs = np.where(mask == 255) # mask에서 흰색 부분의 좌표 찾기
    x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys) # 찾은 좌표의 최소값과 최대값을 가져와서 알약의 왼쪽, 오른쪽, 위, 아래 부분 찾기

    # rect 범위 설정하기
    rect = (max(0, int(width * 0.27)), max(0, int(height * 0.27)), min(width, int(width * 0.5)), min(height, int(height * 0.5)))

    # rect 범위에 따라서 x1, y1, x2, y2 재조정하기
    x1, y1, x2, y2 = max(x1, rect[0]), max(y1, rect[1]), min(x2, rect[0] + rect[2]), min(y2, rect[1] + rect[3])

    # 알약의 경계를 포함해 그 안쪽을 원본 알약 이미지의 색으로 채우기
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # mask를 BGR 색상 공간으로 변환
    result = np.where(mask_color == 255, original_img, 0)  # mask에서 흰색 부분은 원본 이미지의 색으로, 나머지 부분은 검정색으로 채우기

    # 결과 저장하기
    cv2.imwrite('result_image.jpg', result)

    # 이미지 불러오기
    result_img = cv2.imread('result_image.jpg')
    original_img = cv2.imread('cropped_image.jpg')

    # 이미지의 크기를 가져오기
    height, width = original_img.shape[:2]

    # rect 범위 설정하기
    rect = (max(0, int(width * 0.27)), max(0, int(height * 0.27)), min(width, int(width * 0.5)), min(height, int(height * 0.5)))

    # rect 범위 내에서만 result_img 처리하기
    result_img = result_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    # result_img에서 검정색이 아닌 부분의 좌표 찾기
    ys, xs = np.where(np.all(result_img != [0, 0, 0], axis=-1))

    # 찾은 좌표의 최소값과 최대값을 가져와서 알약의 왼쪽, 오른쪽, 위, 아래 부분 찾기
    x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)

    # 원본 이미지에서 해당 영역만 잘라내기
    cropped = original_img[rect[1]+y1:rect[1]+y2, rect[0]+x1:rect[0]+x2]

    # 결과 저장하기
    cv2.imwrite('cropped_result.jpg', cropped)
    front_image = cv2.imread('cropped_result.jpg')
    front_image_rgb = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
    front_image_resized = cv2.resize(front_image_rgb, (128, 128))
    
    # 이미지를 불러오기
    img = Image.open('back.jpg')

    # 이미지 크기 구하기
    width, height = img.size

    # 자를 영역의 크기를 정하기
    crop_size = (1000, 1000)

    # 이미지 가운데를 기준으로 자를 영역을 정하기
    left = (width - crop_size[0])/2
    top = (height - crop_size[1])/2
    right = (width + crop_size[0])/2
    bottom = (height + crop_size[1])/2

    # 이미지를 자르기
    img_cropped = img.crop((left, top, right, bottom))

    # 자른 이미지를 저장하기
    img_cropped.save('cropped_image.jpg')

    # 저장된 이미지를 불러오기
    img = cv2.imread('cropped_image.jpg')
    original_img = img.copy()

    # 이미지의 크기를 가져오기
    height, width = img.shape[:2]

    # Canny edge detection 적용하기
    edges = cv2.Canny(img, threshold1=30, threshold2=100)

    # 가장 밝은 부분만 검출하기 위한 마스크 생성하기
    ret, mask = cv2.threshold(edges, 254, 255, cv2.THRESH_BINARY)

    # mask에서 알약의 왼쪽, 오른쪽, 위, 아래 부분 찾기
    ys, xs = np.where(mask == 255) # mask에서 흰색 부분의 좌표 찾기
    x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys) # 찾은 좌표의 최소값과 최대값을 가져와서 알약의 왼쪽, 오른쪽, 위, 아래 부분 찾기

    # rect 범위 설정하기
    rect = (max(0, int(width * 0.27)), max(0, int(height * 0.27)), min(width, int(width * 0.5)), min(height, int(height * 0.5)))

    # rect 범위에 따라서 x1, y1, x2, y2 재조정하기
    x1, y1, x2, y2 = max(x1, rect[0]), max(y1, rect[1]), min(x2, rect[0] + rect[2]), min(y2, rect[1] + rect[3])

    # 알약의 경계를 포함해 그 안쪽을 원본 알약 이미지의 색으로 채우기
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # mask를 BGR 색상 공간으로 변환
    result = np.where(mask_color == 255, original_img, 0)  # mask에서 흰색 부분은 원본 이미지의 색으로, 나머지 부분은 검정색으로 채우기

    # 결과 저장하기
    cv2.imwrite('result_image.jpg', result)

    # 이미지 불러오기
    result_img = cv2.imread('result_image.jpg')
    original_img = cv2.imread('cropped_image.jpg')

    # 이미지의 크기를 가져오기
    height, width = original_img.shape[:2]

    # rect 범위 설정하기
    rect = (max(0, int(width * 0.27)), max(0, int(height * 0.27)), min(width, int(width * 0.5)), min(height, int(height * 0.5)))

    # rect 범위 내에서만 result_img 처리하기
    result_img = result_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    # result_img에서 검정색이 아닌 부분의 좌표 찾기
    ys, xs = np.where(np.all(result_img != [0, 0, 0], axis=-1))

    # 찾은 좌표의 최소값과 최대값을 가져와서 알약의 왼쪽, 오른쪽, 위, 아래 부분 찾기
    x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)

    # 원본 이미지에서 해당 영역만 잘라내기
    cropped = original_img[rect[1]+y1:rect[1]+y2, rect[0]+x1:rect[0]+x2]

    # 결과 저장하기
    cv2.imwrite('cropped_result.jpg', cropped)
    back_image =  cv2.imread('cropped_result.jpg')
    back_image_rgb = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
    back_image_resized = cv2.resize(back_image_rgb, (128, 128))

    # 두 이미지를 합칩니다.
    merged_image = np.hstack((front_image_resized, back_image_resized))
    merged_image = merged_image / 255.0  # 정규화
    merged_image = np.expand_dims(merged_image, axis=0)  # 모델 예측을 위한 차원 확장

    # 예측을 수행합니다.
    prediction = model.predict(merged_image)

    # Top-5 클래스의 인덱스를 가져옵니다.
    top_5_idx = np.argsort(prediction[0])[-5:]

    # 인덱스를 레이블로 변환합니다.
    top_5_labels = le.inverse_transform(top_5_idx)

    # Top-5 예측 결과를 출력합니다.
    print("Top-5 Predicted classes are:")
    for i, label in enumerate(top_5_labels[::-1], 1):
        print(f"{i}: {label}")

    predicted_item1 = top_5_labels[0].tolist()
    predicted_item2 = top_5_labels[1].tolist()
    predicted_item3 = top_5_labels[2].tolist()
    predicted_item4 = top_5_labels[3].tolist()
    predicted_item5 = top_5_labels[4].tolist()

    return jsonify({
        'result1': predicted_item1,
        'result2': predicted_item2,
        'result3': predicted_item3,
        'result4': predicted_item4,
        'result5': predicted_item5
    })
    

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()