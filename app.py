import numpy as np
from flask import Flask,request,render_template
import pickle
app=Flask(__name__,template_folder='templates')
model=pickle.load(open('D:\SKILL DEVLP\HACKATHONS\GNIT HACK\model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/parkinson')
def parkinson():
    return render_template('parkinson.html')
@app.route('/predict',methods=['POST'])
def predict():
    name = request.form.get('Name')
    gender = request.form.get('gender')
    selected_values = request.form.values()
    converted_values = []
    for value in selected_values:
        try:
            converted_values.append(float(value))
        except ValueError:
            pass
    
    features=[np.array(converted_values)]
    prediction = model.predict(features)   
    output = round(prediction[0],2)
    if(output==0):
      p_result = "parkinson's disease"
      return render_template('output.html',prediction_text=' Test result is negative {}({}) is not effected with {}'.format(name,gender,p_result))
    else:
        p_result = " parkinson's disease !!"
        return render_template('output_p.html',prediction_text=' Test result is postive {}({}) is effected with {}'.format(name,gender,p_result))

if __name__ == '__main__':
    app.run(debug=True)

