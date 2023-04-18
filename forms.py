from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField,FileField,BooleanField
from wtforms.validators import Length, EqualTo, Email, DataRequired
class RegisterForm(FlaskForm):
    username = StringField(label='User Name:', validators=[Length(min=2, max=30), DataRequired()])
    email = StringField(label='Email Address:', validators=[Email(), DataRequired()])
    password = PasswordField(label='Password:', validators=[Length(min=6), DataRequired()])
    confirm_password = PasswordField(label='Confirm Password:', validators=[EqualTo('password'), DataRequired()])
    mobile_number = StringField('Mobile Number', validators=[DataRequired()])
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(),Email()])
    password = PasswordField(label='Password:', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')
class RequestResetForm(FlaskForm):
    email = StringField('Email :', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')
class ResetPasswordForm(FlaskForm):
    password = PasswordField(label='Password:', validators=[DataRequired()])
    confirm_password = PasswordField(label='Confirm Password:', validators=[EqualTo('password'), DataRequired()])
    submit=SubmitField('Reset Password')