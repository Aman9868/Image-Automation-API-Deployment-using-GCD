from app import app
from flask import render_template,request,redirect,url_for, flash,jsonify
import torch
from numba import cuda
from GPUtil import showUtilization as gpu_usage
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)
    print("GPU Usage after emptying the cache")
    gpu_usage()
free_gpu_cache()
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