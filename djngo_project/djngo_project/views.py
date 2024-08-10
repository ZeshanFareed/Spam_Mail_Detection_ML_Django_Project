from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def user(request):
    return render(request , 'userinput.html')


def viewdata(request):
    
    df = pd.read_csv("C:/Users/PMLS/Documents/ML/ML Algorithms/mail_data.csv")
    
    mail_data = df.fillna('')
    
    LB = LabelEncoder()
    mail_data['Category'] = LB.fit_transform(mail_data['Category'])
    
    X = mail_data['Message']
    y = mail_data['Category']
    
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    
    # Initialize the CountVectorizer with correct parameter
    feature_extraction = CountVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    
    
    LR = LogisticRegression()
    LR.fit(X_train_features , y_train)

    
    if request.method == 'GET' and 'Message' in request.GET:
        user_message = request.GET['Message']
        user_message_transformed = feature_extraction.transform([user_message])
    
    # Make prediction
    y_pred = LR.predict(user_message_transformed)
        
    # Determine the label of the prediction
    prediction_label = 'Spam' if y_pred[0] == 1 else 'Ham'
        
    data = {
            'message': 'Your Mail is classified as',
            'prediction': prediction_label
        }
 
 
    return render(request , 'viewdata.html' , data)