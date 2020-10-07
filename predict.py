import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import train_model as tm
import math 

def load_historical_data(start_index = 0, end_index = 0)  :
    historical_data = pd.read_csv('data/training-data.csv')
    historical_data = tm.normalize_data(historical_data)
    
    print('\n\n\nLoaded data')

    if start_index < 0:
        start_index = len(historical_data.index) + start_index
    if end_index < 0:
        end_index = len(historical_data.index) + end_index 
    elif end_index == 0:
        end_index = len(historical_data.index)
        
    return historical_data.loc[start_index:end_index]
    
def predict(features_data,model):
        
    features = {name:np.array(value) for name, value in features_data.items()}
    
    rpms = features.pop('rpms')
    
    predictions = model.predict(features)
    print('Made prediction')
    results = pd.DataFrame()



    for i, prediction in enumerate(predictions):
        results.loc[i,'rpms'] = prediction[0] * 71230170772

    for i, row in features_data.iterrows():
        features_data.loc[i,'rpms'] = row['rpms'] * 71230170772

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
            
            
    axes.plot(historical_data['rpms'][-10:] /1000000000, label='Actual RPMs')
        
    axes.plot(results['rpms'][-10:]/1000000000, label='model')
    plt.legend(loc= "lower left")
    

def predict_historical():

    model = keras.models.load_model('model')
   
    
    historical_data = load_historical_data()

    prediction_data = tm.normalize_data(pd.read_csv('data/prediction-data.csv'))
    historical_data = historical_data.append(prediction_data, ignore_index = True)
    results,historical_data  = predict(historical_data, model)

    plot_results(results, historical_data)
    
    del results['index']
    results.to_csv('data/predicted-rpms.csv', index = False)    

    


#tm.train()
predict_historical()
plt.show()


