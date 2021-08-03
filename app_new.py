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
		else:
		ds=pd.read_csv('stroke_prediction.csv')
		X=ds.drop('target',axis=1)
		y=ds.iloc[:,-1]
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)
		reg=LogisticRegression(max_iter=1200000)
		reg.fit(X_train,y_train)
		Age=int(request.form['Age'])
		gender=int(request.form['gender'])
		cp=int(request.form['cp'])
		trestbps=int(request.form['trestbps'])
		chol=int(request.form['chol'])
		fbs=int(request.form['fbs'])
		restecg=int(request.form['restecg'])
		thalach=int(request.form['thalach'])
		new=np.array([[Age,gender,cp,trestbps,chol,fbs,restecg,thalach,ds['exang'].mean(),ds['oldpeak'].mean(),ds['slope'].mean(),ds['ca'].mean(),ds['thal'].mean()]])
		y_pred=reg.predict(new)
		return render_template("result.html",y_pred=y_pred)
	
@app.route('/heartattack_prediction',methods=['GET','POST'])
def kidney():
	if request.method=='GET':
		return render_template('heart_attack_prediction.html')
	else:
		ds=pd.read_csv('heartattack_prediction.csv')
		X=ds.drop('target',axis=1)
		y=ds.iloc[:,-1]
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)
		reg=LogisticRegression(max_iter=1200000)
		reg.fit(X_train,y_train)
		Age=int(request.form['Age'])
		gender=int(request.form['gender'])
		cp=int(request.form['cp'])
		trestbps=int(request.form['trestbps'])
		chol=int(request.form['chol'])
		fbs=int(request.form['fbs'])
		restecg=int(request.form['restecg'])
		thalach=int(request.form['thalach'])
		new=np.array([[Age,gender,cp,trestbps,chol,fbs,restecg,thalach,ds['exang'].mean(),ds['oldpeak'].mean(),ds['slope'].mean(),ds['ca'].mean(),ds['thal'].mean()]])
		y_pred=reg.predict(new)
		return render_template("result1.html",y_pred=y_pred)

	
@app.route('/')
def index():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
	
## if __name__ == '__main__':
##       app.run(host='0.0.0.0', port=5000)
