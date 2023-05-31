from flask import Flask, request, jsonify
import numpy as np
import json

#Flask 객체 인스턴스 생성
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/test', methods=['POST'])
def test():
    lists = request.args['file_name']
    lists = lists.split(',')
    # data = []
    data = ['가스디알정50밀리그램(디메크로틴산마그네슘)']
    # data = { "name" : '가스디알정50밀리그램(디메크로틴산마그네슘)' }
    # for list in lists:
    #     data.append(list)

    return jsonify({
        'result': data
    })

if __name__ == '__main__':
    app.run()