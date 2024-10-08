import json

from flask import Blueprint, request, Response
from database.database import connection
from http import HTTPStatus

log = Blueprint('log', __name__)

@log.route('/upload', methods=['POST'])
def upload():
    data = request.json
    date = data.get('date')
    time = data.get('time')
    section = data.get('section')
    action = data.get('action')

    try:
        conn = connection()
        cursor = conn.cursor()

        query = 'INSERT INTO log_data () VALUES (%s, %s, %s, %s)'
        cursor.execute(query, (date, time, section, action))
        conn.commit()
        
        cursor.close()
        conn.close()
        return Response(None, status=HTTPStatus.OK)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)

@log.route('/download', methods=['POST'])
def download():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    sections = data.get('sections')
    actions = data.get('actions')

    try:
        conn = connection()
        cursor = conn.cursor()

        parameters = [start_date, end_date]
        query = 'SELECT date, time, section, action FROM log_data WHERE date BETWEEN %s AND %s'
        if len(sections) > 0:
            query += " AND section IN (" + ", ".join(["%s"] * len(sections)) + ")"
            parameters.extend(sections)

        if len(actions) > 0:
            query += " AND action IN (" + ", ".join(["%s"] * len(actions)) + ")"
            parameters.extend(actions)

        cursor.execute(query, parameters)
        result = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return Response(json.dumps(result, default=str), mimetype='application/json', status=HTTPStatus.OK)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)