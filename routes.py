from app import app, db,bcrypt,mail
from flask import render_template,request,redirect,url_for, flash,jsonify
import torch
from numba import cuda
from GPUtil import showUtilization as gpu_usage
from functions import extract_contact_info,extract_airport_codes,first_to_third_person,translate_text
from forms import RegisterForm,LoginForm,RequestResetForm,ResetPasswordForm
from models import User
from flask_login import login_user,logout_user,current_user, login_required
from flask_mail import Message

#### -----------------------CUDA Description--------------------##############################3
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
else:
    print("CUDA is not available")
#################-----------------Gpu Memory Releaser----------------------------####################
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


##-------------------------Homepage--------------------------------------------#####
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
#####---------------------------Register------------------------------------------#####
@app.route('/register',methods=['GET','POST'])
def register_page():
    form=RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email address already exists', 'danger')
            return redirect(url_for('register_page'))
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register_page'))
        #password_hash = generate_password_hash(form.password.data).decode('utf-8')
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user_create = User(username=form.username.data, email=form.email.data, password=hashed_password,mobile_number=form.mobile_number.data)
        db.session.add(user_create)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login_page'))
    if form.errors !={}:
        for err in form.errors.values():
            flash(f'There was an error with creating a user: {err}', category='danger')
    return render_template('register.html',form=form)

#####-----------------------Login Page-----------------------------------#######
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = LoginForm()
    if form.validate_on_submit():
       user = User.query.filter_by(email=form.email.data).first()
       if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('You have been logged in!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('home_page'))
       else:
            flash('Email and password are not match! Please try again', category='danger')

    return render_template('login.html', form=form)
################--------------------Logout Page---------------------------------#######
@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))

############------------------------------RESET PASSWORD------------------------------#############

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)
@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
            flash('An email has been sent with instructions to reset your password.', 'info')
            return redirect(url_for('login'))
        else:
            flash('There is no account with that email. You must register first.', 'warning')
            return redirect(url_for('reset_request'))
    return render_template('reset.html', title='Reset Password', form=form)

@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login_page'))
    return render_template('res_token.html', title='Reset Password', form=form)
############3--------------------Contact Page---------------------------------------################3
@app.route("/contact", methods=['GET', 'POST'])
def contact_page():
    if request.method=="POST":
        name=request.form.get('inputName4')
        email=request.form.get('inputEmail4')
        mobile=request.form.get('inputNumber4')
        message=request.form.get('inputMessage')
        service=request.form.get('inputState')
        msg= Message(subject=f'Mail from {name}',
                     body=f'Name : {name}\nEmail from :{email}\nMobile : {mobile}\nMessage : {message}\nService he wants :{service}',
                     sender=email,recipients=['itsaman9868@gmail.com'])
        mail.send(msg)
        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('contact_page'))
    return render_template('contact.html')

#####------------------------Subscribe Channel----------------------------------#############
@app.route('/subscribe', methods=['POST'])
def subscribe():
    email = request.form.get('email')
    msg = Message('Newsletter Subscription',sender=email ,recipients=['itsaman9868@gmail.com'])
    msg.body = 'Thank you for subscribing to our newsletter!'
    mail.send(msg)
    flash('Thank you for subscribing to our newsletter!', 'success')
    return redirect(url_for('home_page'))
 #####----------------Selection of Text Service Page-------------------------###########
@app.route("/tryitout")
def tryit():
    return render_template('tryitout.html')

############----------------Requesting-------------------------------#############
#############--------------------------Analyze Text--------------------------------------######
@app.route("/analyze")
def analyze():
    finaltext = request.args.get('text', 'default') # text variable request
    removestop = request.args.get('removestop','off') # stopword request
    removepunc = request.args.get('removepunc', 'off') # punctutaion request
    mask = request.args.get('mask', 'off')
    sentiment = request.args.get('sentiment', 'off')
    spell = request.args.get('spell', 'off')
    summary=request.args.get('summary','off') # text summary request
    token = request.args.get('token', 'off')
    generate=request.args.get('generate','off')
    que=request.args.get('que','off')
    person = request.args.get('person', 'off') # third person convert request
    exname=request.args.get('exname','off')
    acode=request.args.get('acode','off')
    sqr=request.args.get('sqr','off')
    grmr=request.args.get('grmr','off') # grammear corrector request
    timage=request.args.get('timage','off')
    vimage=request.args.get('vimage','off')
    trans=request.args.get('trans','off') # Text translation request
    paras=request.args.get('paras','off')
    sde=request.args.get('sde','off')  # Text to speech request
app.run(debug=True)