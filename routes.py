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
from stop_words import get_stop_words
import nltk
from nltk.tokenize import word_tokenize
from diffusers import LDMSuperResolutionPipeline,DPMSolverMultistepScheduler,DiffusionPipeline,StableDiffusionPipeline
import os
import cv2
from textblob import Word
from string import punctuation
from transformers import pipeline,AutoTokenizer, AutoModelForQuestionAnswering
from flair.data import Sentence
from flair.models import SequenceTagger
from PIL import Image, ImageDraw
from keras_cv.models import StableDiffusion
import soundfile as sf
from transformers import AutoModelWithLMHead, AutoTokenizer,SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from parrot import Parrot
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

    ### ---------------------------Stopwords Removal-------------------------------#####
    if removestop == "on":
        stop_words = get_stop_words('en')
        word_tokens = word_tokenize(finaltext)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        result=' '.join([i for i in filtered_sentence])
        params1 = 'Removed Stopwords'
        params2=result
        return render_template('analyze.html',purpose=params1,analyzed_text=params2)
    ######-----------------Text2Video---------------------------------------##############
    elif vimage=="on":
        # load pipeline
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# optimize for GPU memory
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        #generate
        prompt = finaltext
        frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames
        # save frames as video using OpenCV
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

        return render_template('analyze.html', video_path=output_file)
    ####--------------------------------Remove Punctuations------------------------------#####
    elif (removepunc == "on"):
        def strip_punctuation(s):
            return ''.join(c for c in s if c not in punctuation)
        params4 = strip_punctuation(finaltext)
        params3='Remove Punctuations'
        return render_template('analyze.html', purpose=params3, analyzed_text=params4)
    ####------------------------------Spell Checker--------------------------------------#####
    elif (spell == "on"):
        w = Word(finaltext)
        check = w.spellcheck()
        param5 = 'Spell Check'
        params6=check
        return render_template('analyze.html', purpose=param5, analyzed_text=params6)
    ########-------------------------Sentiment Analysis-----------------------------######
    elif (sentiment=="on"):
        sent = pipeline('sentiment-analysis')
        res=sent(finaltext)[0]
        param10=res['label']
        param11='Sentiment Analysis'
        return render_template('analyze.html', purpose=param11, analyzed_text=param10)
    ######------------------------Text Summarization----------------------------------------####
    elif (summary=="on"):
        summary_length = int(request.args.get('summary_length'))
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        sm = summarizer(finaltext, max_length=summary_length, min_length=summary_length-30 ,do_sample=False)[0]['summary_text']
        param7='Summarizer'
        return render_template('analyze.html', purpose=param7, analyzed_text=sm)
    #####---------------- Token Classifictaion----------------------------------------####
    elif (token=="on"):
        def tokenclass(text):
            tagger = SequenceTagger.load("flair/ner-english")  # Load the NER Tagger
            sentence = Sentence(text)  # Make a Sentence
            res = tagger.predict(sentence)  # Run NER over Sentence
            rs = sentence.get_spans('ner')  # Print NER tag
            return rs
        param8='Token Classification'
        param9=tokenclass(finaltext)
        return render_template('analyze.html', purpose=param8, analyzed_text=param9)
    #####------------------------Fill Mask-----------------------------------------#####
    elif (mask=="on"):
        classifier = pipeline("fill-mask")
        items=classifier(finaltext)
        param13='Fill Mask'
        return render_template('analyze2.html', purpose=param13, items=items)
    #######----------------------Text Generation-----------------------------------####
    elif (generate=="on"):
        generator = pipeline('text-generation', model='gpt2')
        param14='Text Generation'
        param15=generator(finaltext,num_return_sequences=3)
        print(param15)
        text_output = []
        for i, text_dict in enumerate(param15):
            line = f" {i + 1}: {text_dict['generated_text'] }"
            text_output.append(line)
        res="".join(text_output)
        return render_template('analyze.html', purpose=param14, analyzed_text=res)
    ######-------------------Third Person Convertor--------------------------------######################
    elif (person=="on"):
        param19='Third Person Convertor'
        param18 = first_to_third_person(finaltext)
        return render_template('analyze.html',purpose=param19,analyzed_text=param18)
    ##############-------------------Text Translation----------------------------###########
    elif (trans=="on"):
        t1="Text Translation"
        t2=translate_text(finaltext)
        return render_template('analyze.html',purpose=t1,analyzed_text=t2)
    #####-------------Contact Inforation Extractor --------------------########
    elif (exname=="on"):
        param22=extract_contact_info(finaltext)
        param23 = 'Extract Contact Information'
        return render_template('analyze.html', purpose=param23, analyzed_text=param22)
    ##############----------------------Airport Code Extractor------------------------##################
    elif (acode=="on"):
        param25 = 'Extract Airport Code'
        param24 = extract_airport_codes(finaltext)
        return render_template('analyze.html', purpose=param25, analyzed_text=param24)
    ###########-------------------Text2 Image-----------------------------------####
    elif (timage=="on"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model=StableDiffusion()
        img=model.text_to_image(finaltext)
        img=Image.fromarray(img[0]).save('static/output/output.jpg')
        return render_template("analyze.html", img_filename='output.jpg')
    ###############------------------Text to Speech----------------------#######
    elif (sde=="on"):
         processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
         model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
         vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
         inputs = processor(text=finaltext, return_tensors="pt")
         # load xvector containing speaker's voice characteristics from a dataset
         embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
         speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
         speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
         speech_np = speech.squeeze().numpy()
         sf.write("static/output/output.wav",speech_np, samplerate=16000) 
         tt=" Text to Speech"
         return render_template('sound.html',audio_path=url_for('static', filename='output/output.wav'))
    ###################-----------------Text Paraphrase-----------------------############3
    elif(paras=="on"):
        t3="Text Paraphrase"
        parrot = Parrot()
        t4=parrot.augment(finaltext)
        print(t4)
        data=[{"text":text,'score':score} for text,score in t4]
        print(data)
        return render_template('yol.html',data=data)
    #######--------------------TEXT 2 SQL GENERATOR -----------------------------------######

    elif (sqr=="on"):
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
        model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
        input_text = "translate English to SQL: %s </s>" % finaltext
        features = tokenizer([input_text], return_tensors='pt')
        output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
        param32=tokenizer.decode(output[0])
        param33 = 'SQL Generator'
        return render_template('analyze.html', purpose=param33, analyzed_text=param32)
    #############------Grammar Correction-----------------------------#########
    elif (grmr=="on"):
        corrector = pipeline(
              'text2text-generation',
              'pszemraj/flan-t5-large-grammar-synthesis',
              )
        results = corrector(finaltext)
        param35 = 'Grammar Corrector'
        param34 = results[0]['generated_text']
        return render_template('analyze.html', purpose=param35, analyzed_text=param34)
    ########-------------------Questionn Answering------------------------------#####
    elif (que=="on"):
        
        # Split the input into question and context
        input_list = finaltext.strip().split(' ', 1)
        question = input_list[0] + ' '
        context = input_list[1]
        model_name = 'distilbert-base-cased-distilled-squad'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
        result = qa_pipeline({
            'question': question,
            'context': context
        })
        answer = result['answer']
        param37="Question Answering"
        print(answer)
        return render_template('analyze.html', purpose=param37, analyzed_text=answer)
    else :
        return "Error"
app.run(debug=True)