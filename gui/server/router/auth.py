import hashlib

from flask import Blueprint, request, Response, jsonify
from database.database import connection
from http import HTTPStatus

auth = Blueprint('auth', __name__)

@auth.route('/signup', methods=['POST'])
def sign_up():
    data = request.json
    name = data.get('user_name')
    id = data.get('user_id')
    password = data.get('user_password')

    conn = connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM user_data WHERE user_id = %s OR user_name = %s"
    cursor.execute(query, (id, name))
    count = cursor.fetchone()[0]

    response = None

    if count > 0:
        response = HTTPStatus.FOUND
    else:
        response = HTTPStatus.NOT_FOUND
        query = "INSERT INTO user_data (user_name, user_id, user_password) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, id, encryption(password)))
        conn.commit()

    cursor.close()
    conn.close()

    return Response(None, status=response)

@auth.route('/signin', methods=['POST'])
def sign_in():
    data = request.json
    id = data.get('user_id')
    password = data.get('user_password')

    conn = connection()
    cursor = conn.cursor()

    query = "SELECT user_name, user_password FROM user_data WHERE user_id = %s"
    cursor.execute(query, (id, ))
    result = cursor.fetchone()

    response = None

    if result[1] == encryption(password):
        response = HTTPStatus.OK
    else:
        response = HTTPStatus.UNAUTHORIZED

    data = {"user_name": result[0]}

    cursor.close()
    conn.close()

    return jsonify(data), response.value

def encryption(password):
    return hashlib.sha256(password.encode()).hexdigest()