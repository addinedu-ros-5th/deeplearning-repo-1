import io

from flask import Blueprint, request, Response, send_file
from database.database import connection
from http import HTTPStatus

image = Blueprint('image', __name__)

@image.route('/upload', methods=['POST'])
def upload():
    date = request.form.get('date')
    file = request.files['image']
    if file:
        image_data = file.read()
        
        try:
            conn = connection()
            cursor = conn.cursor()
            
            query = 'INSERT INTO images_data () VALUES (%s, %s)'
            cursor.execute(query, (date, image_data))
            conn.commit()
            
            cursor.close()
            conn.close()
            return Response(None, status=HTTPStatus.OK)
        except:
            return Response(None, status=HTTPStatus.EXPECTATION_FAILED)
    
@image.route('/download', methods=['POST'])
def download():
    data = request.json
    date = data.get('date')
    
    try:
        conn = connection()
        cursor = conn.cursor()
    
        query = 'SELECT image FROM images_data WHERE date = %s'
        cursor.execute(query, (date, ))
        result = cursor.fetchone()[0]
        return send_file(io.BytesIO(result), mimetype='image/jpeg', as_attachment=True, download_name=result)
    except:
        return Response(None, status=HTTPStatus.EXPECTATION_FAILED)