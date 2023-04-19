from flask import Flask
import os
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
SECRET_KEY=os.urandom(12).hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///D:/TextImage/test.db'
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER']='static/images'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USER')
app.config['MAIL_PASSWORD'] = "zrcondgiumxculbd"
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('EMAIL_USER')
db = SQLAlchemy(app)
bcrypt=Bcrypt(app)
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
mail = Mail(app)
from routes import *