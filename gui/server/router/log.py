import json

from flask import Blueprint, request, Response
from database.database import connection
from http import HTTPStatus

log = Blueprint('log', __name__)


# Log upload
@log.route('/upload', methods=['POST'])
def upload():
    data = request.json
    date = data.get('date')
    time = data.get('time')
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

        query = 'INSERT INTO log_data () VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'
        cursor.execute(query, (date, time, section, start_time, end_time, 
                               video_path, video_width, video_height, action))
        conn.commit()
        
        cursor.close()
        conn.close()
        return Response(None, status=HTTPStatus.OK)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)

# Log download
@log.route('/download', methods=['POST'])
def download():
    try:
        conn = connection()
        cursor = conn.cursor()

        query = 'SELECT * FROM action_data'
        cursor.execute(query)
        result = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return Response(json.dumps(result), mimetype='application/json', status=HTTPStatus.OK)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)