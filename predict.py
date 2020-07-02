import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import train_model as tm
import math 

def load_rpm_data(model,start_index = 0, end_index = 0)  :
    data = tm.get_normalized_data()
    historical_data = pd.read_csv('data\\training_data.csv')
    print('\n\n\nLoaded data')

    if start_index < 0:
        start_index = len(data.index) + start_index
    if end_index < 0:
        end_index = len(data.index) + end_index 
    elif end_index == 0:
        end_index = len(data.index)
        
    return data.loc[start_index:end_index], historical_data.loc[start_index:end_index]

def normalize_prediction_features(baseline,model):
    features = pd.read_csv('data\\scenario1\\features.csv'.format(baseline))

    max_vals = tm.get_max_vals()

    features = tm.apply_inverse(features,model)

    features['Months_After_9/11'] = features['Months_After_9/11'] / max_vals['Months_After_9/11']

 
    features['MSC'] = features['MSC'] / max_vals['MSC']

    features['Unemployement'] = features['Unemployement'] / max_vals['Unemployement']
    
    features['Labor_Force'] = features['Labor_Force'] / max_vals['Labor_Force']

    features['Avg'] = features['Avg']/ max_vals['Avg']
    features['Avgd'] = features['Avgd']/ max_vals['Avgd']

    return features
    
def predict(features_data,model):
    
    max_vals = tm.get_max_vals(model[1])
    model = model[0]
    features = {name:np.array(value) for name, value in features_data.items()}
    date = np.array(features.pop('Date'))

    prediction = model.predict(features)
    print('Made prediction')

    results = pd.DataFrame()
    
    results['Date'] = date    
    #results['RPMs'] = np.exp(prediction * max_vals['RPMs'])
    results['RPMs'] = (prediction * max_vals['RPMs'])
    
    for i, r in results.iterrows():
        results.loc[i,'RPMs'] = max(r['RPMs'],0)
    
    results = results.reset_index()
    return results

def plot_results(results, historical_data):      
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)
    name = 'Model 3'
    axes.set_title('Monthly Domestic RPMs Time Series')
            
    plt.xlabel('Date')
    plt.ylabel('RPMs (billions)')
            
    all_data = results[name]['RPMs'].append(historical_data['RPMs'])#.append(results['Model 3']['RPMs'])
    low = min(all_data)/1000000000
    high = max(all_data)/1000000000
    rng = (high-low)
    axes.set_ylim(int(low-rng*.1), int(high+rng*.1))
        
    frq = 8
    step = max(int(len(results[name]['Date'])/frq),1)
    tck = range(0,len(results[name]['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(results[name]['Date'][i])
    plt.xticks(tck, tck_dates)
            
    axes.plot(historical_data['Date'], historical_data['RPMs']/1000000000, label='Actual RPMs')
    for model_name in results:
        model = results[model_name]
        axes.plot(model['Date'], model['RPMs']/1000000000, label=model_name)
    plt.legend(loc= "lower left")
    
    
def plot_change(results, historical_data):
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)
    name = 'Model 3'

    axes.set_title('Percent Change in RPMs From 2019')
            
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
                   
    frq = 8
    step = max(int(len(results[name]['Date'])/frq),1)
    tck = range(0,len(results[name]['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(results[name]['Date'][i])
    plt.xticks(tck, tck_dates)
            
    for model_name in results:
        model = results[model_name]
        historical_data = historical_data['RPMs'].reset_index()

        model['change'] = model['RPMs'] / historical_data['RPMs']

        axes.plot(model['Date'], model['change'], label=model_name)
    plt.legend(loc= "upper left")

def predict_historical():
    results = {}

    model1 = keras.models.load_model('models/model3')
    rpms_data,historical_data = load_rpm_data(1)
    results['Model 3']  = predict(rpms_data, [model1,1])

    plot_results(results, historical_data)

def predict_recovery():
    results = {}

    model1 = keras.models.load_model('models/model3')
    rpms_data,historical_data = load_rpm_data(1,-5)

    results['Model 3']  = predict(rpms_data, [model1,1])
    prediction_features = normalize_prediction_features('features.csv',1)
    prediction_results = predict(prediction_features, [model1,1])
    results['Model 3'] = results['Model 3'].append(prediction_results).reset_index()
    print(results)
    plot_results(results, historical_data)

def predict_change():
    results = {}

    model1 = keras.models.load_model('models/model3')
    rpms_data,na = load_rpm_data(2,-5)
    na,historical_data = load_rpm_data(2,-17,-5)
    #del rpms_data['MSC']

    results['Model 3']  = predict(rpms_data, [model1,1])
    prediction_features = normalize_prediction_features('features.csv',1)
    prediction_results = predict(prediction_features, [model1,1])
    results['Model 3'] = results['Model 3'].append(prediction_results).reset_index()

    plot_change(results, historical_data)
#predict_historical()
predict_recovery()  
predict_change()
plt.show()
