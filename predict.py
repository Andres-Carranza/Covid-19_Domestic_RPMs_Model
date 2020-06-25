import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import train_model as tm

def load_rpm_data(start_index, end_index)  :
    data = tm.get_normalized_data()
    print('\n\n\nLoaded data')

    if start_index < 0:
        start_index = len(data.index) + start_index
    if end_index < 0:
        end_index = len(data.index) + end_index 
    elif end_index == 0:
        end_index = len(data.index)
        
    return data.loc[start_index:end_index]

    
def predict(features,model):
    max_vals = tm.get_max_vals()

    features = {name:np.array(value) for name, value in features.items()}
    label = np.array(features.pop('RPMs'))
    date = np.array(features.pop('Date'))
    
    prediction = model.predict(features)
    print('Made prediction')

    results = pd.DataFrame()
    
    results['Date'] = date
    
    results['Label'] = label * max_vals['RPMs']
    
    results['Prediction'] = prediction * max_vals['RPMs']
    
    results = results.reset_index()
    return results

def plot_results(results):
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)
    
    axes.set_title('Monthly Domestic RPMs Time Series Model')
        
    plt.xlabel('Date')
    plt.ylabel('RPMs (billions)')
        
    low = min(results['Label'])/1000000000
    high = max(results['Label'])/1000000000
    rng = high-low
    axes.set_ylim(int(low-rng*.1), int(high+rng*.1))
    
    frq = 12
    step = max(int(len(results['Date'])/frq),1)
    tck = range(0,len(results['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(results['Date'][i])
    plt.xticks(tck, tck_dates)
        
    axes.plot(results['Date'], results['Label']/1000000000, label='Actual RPMs')
    axes.plot(results['Date'], results['Prediction']/1000000000, label='Predicted RPMs')
    plt.legend(loc="upper left")
    
    plt.show()
    
model = keras.models.load_model('models/model')

plot_results(predict(load_rpm_data(0,0), model))

