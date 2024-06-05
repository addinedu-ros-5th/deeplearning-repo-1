import json

from flask import Blueprint, request, Response
from database.database import connection
from http import HTTPStatus

log = Blueprint('log', __name__)

@log.route('/upload', methods=['POST'])
def upload():
    data = request.json
    date = data.get('date')
    section = data.get('section')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    video_path = data.get('video_path')
    video_width = data.get('video_width')
    video_height = data.get('video_height')
    action = data.get('action')

    try:
        conn = connection()
        cursor = conn.cursor()

        query = 'INSERT INTO log_data () VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
        cursor.execute(query, (date, section, start_time, end_time, 
                               video_path, video_width, video_height, action))
        conn.commit()
        
        cursor.close()
        conn.close()
        return Response(None, status=HTTPStatus.OK)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)


# 필터 기능 구현 남음
@log.route('/download', methods=['POST'])
def download():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    try:
        conn = connection()
        cursor = conn.cursor()

        query = 'SELECT date, section, start_time, end_time, action FROM log_data WHERE date BETWEEN %s AND %s'
        cursor.execute(query, (start_date, end_date))
        result = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return Response(json.dumps(result, default=str), mimetype='application/json', status=HTTPStatus.OK)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)