# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:08:17 2021

@author: SHUB
"""
import pandas as pd
from flask import Flask, render_template, request
import pickle


df = pd.read_csv("StudentsPerformance.csv")

app = Flask('project')

def features(gender, race, education, lunch, course):
    test_row = [gender, race, education, lunch, course]
    
    cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']
    
    df_cat = df[cat_features]
  
    df_cat.loc[(len(df_cat.index))] = test_row
    
    df_cat_dummy = pd.get_dummies(df_cat,drop_first=True)
    
    input_values = df_cat_dummy.iloc[-1].values
   
    model = pickle.load(open('model.sav', 'rb'))
    
    pred_values = model.predict([input_values])
    
    return pred_values

@app.route('/') 
def show_form():
    return render_template('predictor_form.html')

@app.route('/visualisation', methods=["POST"])
def visualisation():
    return render_template('Report.html')

@app.route('/result', methods=["POST"])
def results():
    form = request.form
    if request.method == 'POST':
       
        gender = request.form['gender']
        race = request.form['race']
        education = request.form['education']
        lunch = request.form['lunch']
        course = request.form['course']
       
        pred_values = features(gender, race, education, lunch, course)
        pred_values = pred_values[0]
        math_score = round(pred_values[0],0)
        reading_score = round(pred_values[1],0)
        writing_score = round(pred_values[2],0)
        
        return render_template('result_form.html', gender=gender, race = race, education=education, 
                               lunch=lunch, course=course ,math_score=math_score, reading_score=reading_score,
                               writing_score=writing_score)



app.run(debug=True)
