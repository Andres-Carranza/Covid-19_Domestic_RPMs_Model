import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import train_model as tm

def load_rpm_data(start_index = 0, end_index = 0)  :
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

def normalize_prediction_features(baseline):
    features = pd.read_csv('data\\prediction_features\\{}.csv'.format(baseline))

    max_vals = tm.get_max_vals()

    features = tm.apply_inverse(features)

    
    features['Months_After_Covid-19'] = features['Months_After_Covid-19'] / max_vals['Months_After_Covid-19']
    features['Months_After_9/11'] = features['Months_After_9/11'] / max_vals['Months_After_9/11']

    features['CPG'] = features['CPG'] / max_vals['CPG']
 
    features['LCC_Market_Share'] = features['LCC_Market_Share'] / max_vals['LCC_Market_Share']

    features['Unemployement'] = features['Unemployement'] / max_vals['Unemployement']
    
    features['Labor_Force'] = features['Labor_Force'] / max_vals['Labor_Force']
    
    return features
    
def predict(features_data,model):
    max_vals = tm.get_max_vals()

    features = {name:np.array(value) for name, value in features_data.items()}
    date = np.array(features.pop('Date'))
    
    prediction = model.predict(features)
    print('Made prediction')

    results = pd.DataFrame()
    
    results['Date'] = date    
    results['RPMs'] = prediction * max_vals['RPMs']
    
    results = results.reset_index()
    return results

def plot_results(results, historical_data):
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)
    
    axes.set_title('Monthly Domestic RPMs Time Series Model')
        
    plt.xlabel('Date')
    plt.ylabel('RPMs (billions)')
        
    low = min(results['RPMs'])/1000000000
    high = max(results['RPMs'])/1000000000
    rng = high-low
    axes.set_ylim(int(low-rng*.1), int(high+rng*.1))
    
    frq = 10
    step = max(int(len(results['Date'])/frq),1)
    tck = range(0,len(results['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(results['Date'][i])
    plt.xticks(tck, tck_dates)
        
    axes.plot(historical_data['Date'], historical_data['RPMs']/1000000000, label='Actual RPMs')
    axes.plot(results['Date'], results['RPMs']/1000000000, label='Predicted RPMs')
    plt.legend(loc="upper left")
    

    
model = keras.models.load_model('models/model1')

rpms_data,historical_data = load_rpm_data(0,0)
results = predict(rpms_data, model)

'''
prediction_features = normalize_prediction_features('current')
prediction_results = predict(prediction_features, model)
plot_results(results.append(prediction_results).reset_index(), historical_data)

prediction_features = normalize_prediction_features('2019')
prediction_results = predict(prediction_features, model)
plot_results(results.append(prediction_results).reset_index(), historical_data)
'''
plot_results(results, historical_data)
plt.show()