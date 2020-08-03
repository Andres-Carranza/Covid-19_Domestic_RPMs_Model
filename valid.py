import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import math 
from datetime import datetime as dt

def apply_inverse(data):
    data['months-since-covid-19'] = 1/ data['months-since-covid-19'] ** 1
    data['months-since-9/11'] = 1 / data['months-since-9/11'] **2
    data = data.replace(math.inf,0)

    return data

def normalize_data(data):
    data = apply_inverse(data)
    
    for col in data.columns:
        data[col] = data[col] / max(max(data[col]),1)

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



def predict_model(model_name):

    model = keras.models.load_model('models/{}'.format(model_name))

    prediction_data = normalize_data(pd.read_csv('data/validation/validation_data.csv'))
    print(pd.read_csv('data/validation/validation_data.csv'))
    #prediction_data = prediction_data.drop(range(threshold, len(prediction_data)))
    
    max_rpms = 71230170772

    results = predict(prediction_data, model,max_rpms)

    return results 




def update_prediction():
    model_names = ['baseline','pessimistic','optimistic']


    for model_name in model_names:

        predictions = predict_model(model_name)
        print('made_prediction for: {}'.format(model_name))
        predictions.to_csv('data/validation/{}.csv'.format(model_name),index = False)



update_prediction()