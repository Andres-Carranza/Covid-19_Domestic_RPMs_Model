  
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import math 
from datetime import datetime as dt

def apply_inverse(data):
    data['months-since-covid-19'] = 1/ data['months-since-covid-19'] ** 2
    data['months-since-9/11'] = 1 / data['months-since-9/11'] **2
    data = data.replace(math.inf,0)

    return data

def normalize_data(data):
    data = apply_inverse(data)

    max_vals = {
        'jan':1,'feb':1,'mar':1,'apr':1,'may':1,'jun':1,'jul':1,'aug':1,'sep':1,'oct':1,'nov':1,'dec':1,
        'leap-feb':1,'thanksgiving-nov':1,'thanksgiving-dec':1,
        'gulf-war':1, '9/2001':1, 'iraq-war':1,'sars-outbreak':1,'great-recession':1,
        'months-since-9/11':1,'months-since-covid-19':1,
        'unemployment-rate':14.7,'nonfarm-payroll':152463,
        'deaths':1955.033333,'new-infected':204180.4,'current-infected':3367528.167,
        'rpms':71230170772,
    }
    for col in data.columns:
        data[col] = data[col] / max_vals[col]

    return data

def predict(features_data,model,max_rpms):
        
    features = {name:np.array(value) for name, value in features_data.items()}
        
    predictions = model.predict(features)
    results = pd.DataFrame()


    for i, prediction in enumerate(predictions):
        results.loc[i,'rpms'] = prediction[0] * max_rpms

    for i, row in results.iterrows():
        results.loc[i,'rpms'] = max(row['rpms'],0)
   
    return results



def predict_model():

    model = keras.models.load_model('model')
    prediction_data = pd.read_csv('data/validation-data.csv')
    prediction_data.pop('rpms')
    prediction_data = normalize_data(prediction_data)
    
    max_rpms = 71230170772

    results = predict(prediction_data, model,max_rpms)

    return results 




def update_prediction():

    predictions = predict_model()
    print('made_prediction for')
    predictions.to_csv('data/validation-predictions.csv',index = False)



update_prediction()
