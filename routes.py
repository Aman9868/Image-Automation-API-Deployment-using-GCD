import panopticapi
from panopticapi.utils import id2rgb, rgb2id
import torch
from app import app, db
torch.cuda.is_available()
import io
from functions import extract_contact_info,extract_airport_codes
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from transformers import DetrFeatureExtractor
from transformers import DetrForSegmentation
from diffusers import LDMSuperResolutionPipeline
from transformers import ViltProcessor, ViltForQuestionAnswering
import tensorflow as tf
from transformers import pipeline
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog
from copy import deepcopy
from huggingface_hub import from_pretrained_keras
import json
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import render_template,request,redirect,url_for, flash,jsonify
from string import punctuation
from textblob import Word
from forms import RegisterForm,LoginForm
from models import User
from flair.data import Sentence
from flair.models import SequenceTagger
import cv2
from flask_login import login_user,logout_user
import requests
from PIL import Image, ImageDraw
import os
from werkzeug.utils import secure_filename
import numpy as np
import openai
##-------------------------Homepage--------------------------------------------#####
@app.route("/")
@app.route("/home")
def home_page():
    return render_template('index.html')


#####---------------------------Register------------------------------------------#####
@app.route('/register',methods=['GET','POST'])
def register_page():
    form=RegisterForm()
    if form.validate_on_submit():
        user_create=User(username=form.username.data,email_address=form.email_address.data,
                         password=form.password1.data)
        db.session.add(user_create)
        db.session.commit()
        return redirect(url_for('login_page'))
    if form.errors !={}:
        for err in form.errors.values():
            flash(f'There was an error with creating a user: {err}', category='danger')
    return render_template('register.html',form=form)

#####-----------------------Login Page-----------------------------------#######
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('home_page'))
        else:
            flash('Username and password are not match! Please try again', category='danger')

    return render_template('login.html', form=form)
################--------------------Logout Page---------------------------------#######
@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))

@app.route('/base')
def base_page():
    return render_template('base.html')
##########----------------------------Home-------------------------------------------------------###

@app.route("/service")
def service_page():
    return render_template("service.html")
@app.route("/contact")
def contact_page():
    return render_template('contact.html')
@app.route("/tryitout")
def tryit():
    return render_template('tryitout.html')
@app.route("/projects")
def project_page():
    return render_template('projects.html')
@app.route("/about")
def about_page():
    return render_template('about.html')

#-------------------------------Conatct Info extractor----------------------------#


#############--------------------------Analyze Text--------------------------------------######
@app.route("/analyze")
def analyze():
    finaltext = request.args.get('text', 'default')
    removestop = request.args.get('removestop','off')
    removepunc = request.args.get('removepunc', 'off')
    mask = request.args.get('mask', 'off')
    sentiment = request.args.get('sentiment', 'off')
    spell = request.args.get('spell', 'off')
    summary=request.args.get('summary','off')
    token = request.args.get('token', 'off')
    generate=request.args.get('generate','off')
    spread=request.args.get('spread','off')
    person = request.args.get('person', 'off')
    kword=request.args.get('kword','off')
    exname=request.args.get('exname','off')
    acode=request.args.get('acode','off')
    eoutline=request.args.get('eoutline','off')
    pth=request.args.get('pth','off')
    idea=request.args.get('idea','off')
    sqr=request.args.get('sqr','off')
    grmr=request.args.get('grmr','off')
    timage=request.args.get('timage','off')


### ---------------------------Stopwords Removal-------------------------------#####
    if removestop == "on":
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(finaltext)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        result=' '.join([i for i in filtered_sentence])
        params1 = 'Removed Stopwords'
        params2=result
        return render_template('analyze.html',purpose=params1,analyzed_text=params2)

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
######------------------------Text Summarization----------------------------------------####
    elif (summary=="on"):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        sm=summarizer(finaltext,max_length=130, min_length=30, do_sample=False)[0]
        sm=sm['summary_text']
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
########-------------------------Sentiment Analysis-----------------------------######
    elif (sentiment=="on"):
        sent = pipeline('sentiment-analysis')
        res=sent(finaltext)[0]
        param10=res['label']
        param11='Sentiment Analysis'
        return render_template('analyze.html', purpose=param11, analyzed_text=param10)

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
#######------------------- Text To Sheet -------------------------------------#######
    elif (spread=="on"):
        openai.api_key ="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"
        param16 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0.5,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        param16=param16['choices'][0]['text']
        param17='Text 2 Spreadsheet'
        param16=param16.replace('Movie | Year of Release', '')
        #param16=param16.translate({ord(i): None for i in '|'})
        return render_template('analyze.html',purpose=param17,analyzed_text=param16)

######-------------------Third Person Convertor--------------------------------######################

    elif (person=="on"):
        openai.api_key = "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"
        param18=openai.Completion.create(
                model="text-davinci-003",
                prompt=finaltext,
                temperature=0,
                max_tokens=60,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
                )
        param19='Third Person Convertor'
        param18 = param18['choices'][0]['text']
        return render_template('analyze.html',purpose=param19,analyzed_text=param18)

####--------------Keyword Extractor---------------------------------########
    elif (kword=="on"):
        openai.api_key = "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"
        param20 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        param20 = param20['choices'][0]['text']
        param21 = 'Keyword Extractor'
        return render_template('analyze.html', purpose=param21, analyzed_text=param20)
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
###-------------------------Essay Outliner-------------------------------########
    elif (eoutline=="on"):
        openai.api_key = "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"
        param26 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        param27 = 'Essay Outliner'
        param26 = param26['choices'][0]['text']
        return render_template('analyze.html', purpose=param27, analyzed_text=param26)
######---------------------Code To Language Convertor----------------------------#########
    elif (pth=="on"):
        openai.api_key = "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"
        param28 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        param29 = 'Code to Language'
        param28 = param28['choices'][0]['text']
        return render_template('analyze.html', purpose=param29, analyzed_text=param28)
####----------------Idea Generator-------------------------------#####
    elif (idea=="on"):
        openai.api_key = "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"
        param30 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        param31 = 'Idea Generator'
        param30 = param30['choices'][0]['text']
        return render_template('analyze.html', purpose=param31, analyzed_text=param30)

#######--------------------TEXT 2 SQL GENERATOR -----------------------------------######

    elif (sqr=="on"):
        openai.api_key = "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"
        param32 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        param33 = 'SQL Generator'
        param32 = param32['choices'][0]['text']
        return render_template('analyze.html', purpose=param33, analyzed_text=param32)
#############------Grammar Correction-----------------------------#########
    elif (grmr=="on"):
        openai.api_key = "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"
        param34 = openai.Completion.create(
            model="text-davinci-003",
            prompt=finaltext,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        param35 = 'Grammar Corrector'
        param34 = param34['choices'][0]['text']
        return render_template('analyze.html', purpose=param35, analyzed_text=param34)
###########-------------------Text2 Image-----------------------------------####
    elif (timage=="on"):
        api_key=openai.api_key = "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"
        response = requests.post(
            'https://api.openai.com/v1/images/generations',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
            json={
                "model": "image-alpha-001",
                "prompt": finaltext,
                "num_images": 1,
                "size": "1024x1024",
                "response_format": "url"
            }
        )
        response_data = json.loads(response.text)
        print(response_data)
        image_url = response_data['data'][0]['url']
        return render_template("analyze.html", image_url=image_url)

    else :
        return '''Error'''
#@##---------------------------Image Section########################################------------------------

#------------------------------------------Upload Images--------------------------------------------------------------#

app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER']='static/images'
app.config['OUTPUT_FOLDER'] = 'static/output'


# Allow file
def allow_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


## Select Image
@app.route('/choose')
def idx():
    return render_template('im.html')

# Upload image inside directory & move to next page
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allow_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for("display_image", filename=filename))
    else:
        return jsonify(success=False, error="File format not supported"), 400


## Show Uploaded Image & Switch bar Buttons
@app.route("/display/<filename>")
def display_image(filename):
    return render_template("display.html", filename=filename)

# initialize KerasHub for image enhancement

### Do Prediction For Choices
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], data["file_path"])
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], data['file_path'])
    model_type = data["model_type"]

#######----------------Image Classification-----------------------##############

    if model_type == "classification":
        model = pipeline("image-classification")
        result = model(file_path)
        labels = [r["label"] for r in result]
        probs = [r["score"] for r in result]
        return render_template("predict.html", model_type=model_type, result=zip(labels, probs))


##########-------------Image Segmentation-------------------------------##########

    elif model_type == "semantic_seg":
        segmenter = pipeline("image-segmentation")
        output = segmenter(file_path)
        output_image = output[0]['mask']
        output_file = file_path.replace("images", "output")
        output_image.save(output_file)
        return render_template("predict.html", model_type=model_type,result=output_file)


###############---------------Object Detection--------------------#################

    elif model_type == "obdetect":
        im = cv2.imread(file_path)
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = out.get_image()
        cv2.imwrite(output_file,output_image)
        return render_template('yol.html',result=output_file)
#######-------------------LDM Super Resolution------------------------------------------#################
    elif model_type =="imart":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "CompVis/ldm-super-resolution-4x-openimages"
        ss = LDMSuperResolutionPipeline.from_pretrained(model_id)
        img=Image.open(file_path)
        low_res_img = img.resize((128, 128))
        out = ss(low_res_img, num_inference_steps=100, eta=1).images[0]
        out.save(output_file)
        return render_template("predict.html", model_type=model_type, result=output_file)

##################---------Low Light Image Enhancement-------------------------###################
    elif model_type =="lowlight":
        model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)
        # load image and preprocess it for MIRNet
        img = tf.keras.utils.load_img(file_path)
        imgs=img.resize((256, 256), Image.NEAREST)
        input_arr = tf.keras.utils.img_to_array(imgs)
        image = input_arr.astype('float32') / 255.0
        img_tensr = np.expand_dims(image, axis=0)

        # enhance image using MIRNet
        output = model.predict(img_tensr)
        print(f'Type of result: {type(output)}')
        print(f'Structure of result: {output}')
        output_image = tf.keras.utils.array_to_img(output[0])
        output_file = file_path.replace("images", "output")
        output_image.save(output_file)
        return render_template("predict.html", model_type=model_type, result=output_file)
#######--------------Image 2 Text----------------------------------------------###########

    elif model_type =="imgtext":
        ts = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        res = ts(file_path)[0]
        res=res['generated_text']
        return render_template('analyze.html', analyzed_text=res)

#############--------Image Blur--------------------------------------------###########
    elif model_type =="imblur":
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (45, 45), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.merge((thresh, thresh, thresh))
        image = image * thresh
        cv2.imwrite(output_file, image)
        return render_template("predict.html",model_type=model_type,result=output_file)


#########---------------Key Point Detection---------------------------------###################


    elif model_type =="kdt":
        torch._C._cuda_init()
        im=cv2.imread(file_path)
        cfg = get_cfg()  # get a fresh new config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v= v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = v.get_image()
        cv2.imwrite(output_file, output_image)
        return render_template("predict.html", model_type=model_type, result=output_file)


############-----------------DEpth Estimation--------------------------------------##############
    elif model_type=="dpt":
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        cv2.imwrite(output_file,output)
        return render_template("predict.html", model_type=model_type, result=output_file)
####-------------Panoptic Segmentation----------------------------#######
    elif model_type =="oldify":
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        img = Image.open(file_path)
        encoding = feature_extractor(img, return_tensors="pt")
        model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
        outputs = model(**encoding)
        processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
        # We extract the segments info and the panoptic result from DETR's prediction
        segments_info = deepcopy(result["segments_info"])
        # Panoptic predictions are stored in a special format png
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))
        final_w, final_h = panoptic_seg.size
        # We convert the png into an segment id map
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

        # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
        for i in range(len(segments_info)):
            c = segments_info[i]["category_id"]
            segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i][
                "isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

        # Finally we visualize the prediction
        v = Visualizer(np.array(img.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
        v._default_font_size = 20
        v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
        output_image=v.get_image()
        #output_image = Image.fromarray(v.get_image())
        cv2.imwrite(output_file,output_image)
        return render_template("predict.html", model_type=model_type, result=output_file)
    else:
        return "Invalid Model Type"

#################-----------Visual Question Answring-----------------------##
@app.route("/trynew",methods=["GET", "POST"])

def trynew():
    if request.method == "POST":
        image = request.files["image"]
        image=Image.open(image)
        vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        text = request.form["question"]
        # prepare inputs
        encoding = vilt_processor(image, text, return_tensors="pt")
        # forward pass
        outputs = vilt_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer=vilt_model.config.id2label[idx]
        ps="Visual Question Answering"
        return render_template("answer.html", answer=answer,purpose=ps)
    return render_template("vqa.html")



@app.route("/predict_image/<result_path>")
def predict_image(result_path):
    return render_template("predict.html", result_path=result_path)

app.run(debug=True)