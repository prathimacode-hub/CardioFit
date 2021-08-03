from flask import Flask,request,render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

#pip freeze > requirements.txt
app=Flask(__name__)

@app.route('/stroke_prediction',methods=['GET','POST'])
def stroke():
	if request.method=='GET':
		return render_template('stroke_prediction_form.html')
	else:
		with open('stroke_prediction','rb') as f:
			model=pickle.load(f)
		Age=int(request.form['Age'])
		Blood_Pressure=int(request.form['Blood_Pressure'])
		Specific_Gravity=float(request.form['Specific_Gravity'])
		Albumin=int(request.form['Albumin'])
		Sugar=int(request.form['Sugar'])
		Red_Blood_Cells=int(request.form['Red_Blood_Cells'])
		new=np.array([[Age,Blood_Pressure,Specific_Gravity,Albumin,Sugar,Red_Blood_Cells]])
		y_pred=model.predict(new)
		return render_template('result.html',y_pred=y_pred)
	
@app.route('/heartattack_prediction',methods=['GET','POST'])
def kidney():
	if request.method=='GET':
		return render_template('heart_attack_prediction.html')
	else:
		with open('heartatack_prediction','rb') as f:
			model=pickle.load(f)
		Age=int(request.form['Age'])
		Blood_Pressure=int(request.form['Blood_Pressure'])
		Specific_Gravity=float(request.form['Specific_Gravity'])
		Albumin=int(request.form['Albumin'])
		Sugar=int(request.form['Sugar'])
		Red_Blood_Cells=int(request.form['Red_Blood_Cells'])
		new=np.array([[Age,Blood_Pressure,Specific_Gravity,Albumin,Sugar,Red_Blood_Cells]])
		y_pred=model.predict(new)
		return render_template('result1.html',y_pred=y_pred)

@app.route('/')
def index():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
	
## if __name__ == '__main__':
##       app.run(host='0.0.0.0', port=5000)
