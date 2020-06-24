import tensorflow as tf
from tensorflow import keras
import train_model as tm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

model = keras.models.load_model('models/simple_model')

max_vals, data = tm.get_normalized_data()

print('Loaded data\n\n\n')


label_name = 'RPMs'

features = {name:np.array(value) for name, value in data.items()}
label = np.array(features.pop(label_name))


prediction = model.predict(features)

results = pd.DataFrame()
results['Date'] = pd.read_csv('data\\dummy_vars.csv')['Date'][:]
results['Label'] = (label * max_vals['RPMs'])[:]
results['Prediction'] = (prediction * max_vals['RPMs'])[:]
 
results = results.reset_index()   

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