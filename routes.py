from app import app
from flask import render_template,request,redirect,url_for, flash,jsonify
@app.route("/")
@app.route("/home")
def home_page():
    return render_template('index.html')
@app.route("/projects")
def project_page():
    return render_template('projects.html')