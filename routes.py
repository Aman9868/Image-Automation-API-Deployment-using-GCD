from app import app
from flask import render_template,request,redirect,url_for, flash,jsonify
@app.route("/")
@app.route("/home")
def home_page():
    return render_template('index.html')
@app.route("/projects")
def project_page():
    return render_template('projects.html')
@app.route("/about")
def about_page():
    return render_template('about.html')
@app.route('/base')
def base_page():
    return render_template('base.html')