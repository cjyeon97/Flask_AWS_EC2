import os

from flask import Flask, request, redirect, url_for
from flask import send_from_directory
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename

from flask_pymongo import PyMongo
from pymongo import MongoClient
from pymongo.cursor import CursorType

from STD.detection import Detection
from STR.recognition import Recognition

# 업로드한 이미지의 저장 경로
UPLOAD_FOLDER = './flask_upload'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


ALLOWED_EXTENSIONS = set(['png', 'jpg'])

app= Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config["MONGO_URI"] = "mongodb://localhost:27017/HelloWorld"
mongo = PyMongo(app)

api = Api(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# STR할 이미지 업로드
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))
            return redirect('/detection')
        
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>OCR할 이미지 업로드</h1>
    <form action="" method=post enctype=multipart/form-data>
        <p><input type=file name=file>
            <input type=submit value=Upload>
    </form>
    """
# 업로드한 이미지 띄우기
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

# DB에서 검색
@app.route('/searchDB', methods = ['POST', 'GET'])
def result():
    # STR결과 txt파일로 읽기
    log = open(f'flask_upload/ocr_result.txt', 'r', encoding='cp949')
    # 큰 text 3개 저장
    prdt_list = log.read().split('\n')
    log.close()

    # 마지막줄 \n 삭제
    del prdt_list[-1]

    # UPLOAD_FOLDER내 파일 삭제
    for file in os.scandir(UPLOAD_FOLDER):
        os.remove(file.path)

    # DB연결
    user_collection = mongo.db.ocrDB
    min_length = 10000

    i = 1
    str_data = ''
    for prdt_name in prdt_list:
        list_data = list(user_collection.find({"prdt_nm" :{"$regex": prdt_name}}))
        data_length = len(list_data)

        if data_length == 0:
            # 검색값이 없으면
            continue
        else:
            if len(list_data) < min_length:
                min_length = data_length
                str_data = ''
                for dict_data in list_data:
                    for key, value in dict_data.items():
                        str_data += str(key)
                        str_data +=': '
                        str_data += str(value)
                        str_data += '<br/>' 
                    str_data += '===============================================<br/>'               
    if len(str_data)>0:
        return str_data
    else:
        return '검색 결과 없음'
    # return f'<h5>Product Name: { data["prdt_nm"]} <br> Action Date: {data["action_de"] } <br> Extra Info: {data["etc_info"] } <br> Manufacturer: { data["mnfctur_nm"]} <br> Product Class: { data["prdtarm"]} <br> Violence: {data["violt_cn"] } <br> Action: {data["action_cn"] }</h5>'

api.add_resource(Detection, '/detection')
api.add_resource(Recognition, '/recognition')   

if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=8000, debug=True)