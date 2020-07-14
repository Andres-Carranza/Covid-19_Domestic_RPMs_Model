import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import train_model as tm
import math 

def load_historical_data(model_name,start_index = 0, end_index = 0)  :
    historical_data = pd.read_csv('data/{}/training-data.csv'.format(model_name))
    max_rpms = max(historical_data['rpms'])
    historical_data = tm.normalize_data(historical_data)
    
    print('\n\n\nLoaded data')

    if start_index < 0:
        start_index = len(historical_data.index) + start_index
    if end_index < 0:
        end_index = len(historical_data.index) + end_index 
    elif end_index == 0:
        end_index = len(historical_data.index)
        
    return historical_data.loc[start_index:end_index], max_rpms
    
def predict(features_data,model,max_rpms):
        
    features = {name:np.array(value) for name, value in features_data.items()}
    
    rpms = features.pop('rpms')
    
    predictions = model.predict(features)
    print('Made prediction')
    results = pd.DataFrame()



    for i, prediction in enumerate(predictions):
        results.loc[i,'rpms'] = prediction[0] * max_rpms

    for i, row in features_data.iterrows():
        features_data.loc[i,'rpms'] = row['rpms'] * max_rpms

    for i, row in results.iterrows():
        results.loc[i,'rpms'] = max(row['rpms'],0)
   
    results = results.reset_index()
    return results, features_data

def plot_results(results, historical_data):      
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Monthly Domestic RPMs Time Series')
            
    plt.xlabel('Date')
    plt.ylabel('RPMs (billions)')
            
    all_data = results['rpms'].append(historical_data['rpms'])
            
    axes.plot(historical_data['rpms']/1000000000, label='Actual RPMs')
        
    axes.plot(results['rpms']/1000000000, label=model_name)
    plt.legend(loc= "lower left")
    

def predict_historical(model_name):

    model = keras.models.load_model('models/{}'.format(model_name))
   
    
    historical_data,max_rpms = load_historical_data(model_name)


    prediction_data = tm.normalize_data(pd.read_csv('data/{}/prediction-data.csv'.format(model_name)))
    historical_data = historical_data.append(prediction_data, ignore_index = True)
    
    results,historical_data  = predict(historical_data, model,max_rpms)

    plot_results(results, historical_data)
    
    del results['index']
    results.to_csv('data/{}/predicted-rpms.csv'.format(model_name), index = False)    

    
model_names = ['baseline', 'pessimistic','optimistic']
model_name = model_names[0]

tm.train(model_name)
predict_historical(model_name)
plt.show()


