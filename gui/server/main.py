from flask import Flask
from router.auth import auth
from router.log import log

app = Flask(__name__)

app.register_blueprint(auth, url_prefix='/auth')
app.register_blueprint(log, url_prefix='/log')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)