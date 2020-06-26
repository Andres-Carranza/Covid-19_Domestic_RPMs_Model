import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from matplotlib import pyplot as plt
import math
from datetime import datetime
import shutil
def load_data():
    model_data = pd.read_csv('data\\raw\\model_data.csv')
    
    dummy_vars = pd.read_csv('data\\dummy_vars.csv')
    del dummy_vars['Date']
    
    for dummy_var in dummy_vars.head():
        model_data[dummy_var] = dummy_vars[dummy_var]
    
    
    model_data['CPG'] = pd.read_csv('data\\domestic_cpg.csv')['CPG']
    
    rpms = pd.read_csv('data\\domestic_rpms.csv')
    model_data['LCC_Market_Share'] = rpms['Domestic LCC RPMs'] / rpms['Total Domestic RPMs']
    
    model_data['Unemployement']= pd.read_csv('data\\unemployement.csv')['Unemployement']
    model_data['Labor_Force']= pd.read_csv('data\\labor_force.csv')['Labor Force']
    
    model_data['RPMs'] = rpms['Total Domestic RPMs']
    
    model_data.to_csv('model_data.csv', index = False)

def apply_inverse(training_data,model):
    #Inverse squares of normalized data
    training_data['Months_After_9/11'] = 1 / training_data['Months_After_9/11'] **2
    training_data = training_data.replace(math.inf,0)

    return training_data

def get_normalized_data(model = 1):
    training_data = pd.read_csv('data\\training_data.csv')
    

    training_data = apply_inverse(training_data,model)

    training_data['Months_After_9/11'] = training_data['Months_After_9/11'] / max(training_data['Months_After_9/11'])

    training_data['CPG'] = training_data['CPG'] / max(training_data['CPG'])
 
    training_data['LCC_Market_Share'] = training_data['LCC_Market_Share'] / max(training_data['LCC_Market_Share'])

    training_data['Unemployement'] = training_data['Unemployement'] / max(training_data['Unemployement'])
    
    training_data['Labor_Force'] = training_data['Labor_Force'] / max(training_data['Labor_Force'])
    
    training_data['Avg'] = np.log(training_data['Avg'] )
    training_data['Avgd'] = np.log(training_data['Avgd'] )
    training_data = training_data.replace(-math.inf,0)

    training_data['Avg'] = training_data['Avg']/ max(training_data['Avg'])
    training_data['Avgd'] = training_data['Avgd']/ max(training_data['Avgd'])

    training_data['RPMs'] = np.log(training_data['RPMs'] )
    training_data['RPMs'] = training_data['RPMs'] / max(training_data['RPMs'])

    return training_data

def get_max_vals(model =1):
    training_data = pd.read_csv('data\\training_data.csv')

    training_data['Avg'] = np.log(training_data['Avg'] )
    training_data['Avgd'] = np.log(training_data['Avgd'] )
    training_data = training_data.replace(-math.inf,0)
    
    training_data = apply_inverse(training_data,model)
    training_data['RPMs'] = np.log(training_data['RPMs'] )

    return { 
                'Months_After_9/11': max(training_data['Months_After_9/11']),
                'CPG': max(training_data['CPG']),
                'LCC_Market_Share': max(training_data['LCC_Market_Share']),
                'Unemployement': max(training_data['Unemployement']),
                'Labor_Force': max(training_data['Labor_Force']),
                'RPMs': max(training_data['RPMs']),
                'Avg':max(training_data['Avg']),
                'Avgd':max(training_data['Avgd'])}    
def get_feature_layer(data):
    feature_columns = []
    for column in data.columns:
        if column in['RPMs','Date','LCC_Market_Share'] :
            continue
        feature_columns.append(tf.feature_column.numeric_column(column))
    return tf.keras.layers.DenseFeatures(feature_columns)

def plot_the_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max() * 1.03])
    plt.show()  

def create_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    
    model.add(feature_layer)
    
    reg_rate = 0.00
    dropout_rate= 0.0
    model.add(tf.keras.layers.Dense(units=300,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))


    model.add(tf.keras.layers.Dense(units=300,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    
    model.add(tf.keras.layers.Dense(units=128,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))    
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    model.add(tf.keras.layers.Dense(units=128,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))


    model.add(tf.keras.layers.Dense(units=128,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,  
                                  name='Output'))                              
  
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

def train_model(model, dataset, epochs, label_name, batch_size=None):
    # Split the dataset into features and label.
    del dataset['Date']
    del dataset['LCC_Market_Share']
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    
    history = model.fit(x=features, y=label, batch_size=batch_size,epochs=epochs) 
    
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
      
    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch. 
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    
    return epochs, mse

def train():
    training_data = get_normalized_data()
    print('Normalized Data')
    
    feature_layer = get_feature_layer(training_data)
    print('Made feature layer')
    
    #The following variables are the hyperparameters.
    learning_rate = 0.01
    epochs = 100
    batch_size = None
    
    model = create_model(learning_rate, feature_layer)
    print('Model created')
    
    # Specify the label
    label_name = "RPMs"
    
    # Train the model on the normalized training set. We're passing the entire
    # normalized training set, but the model will only use the features
    # defined by the feature_layer.
    epochs, mse = train_model(model,training_data , epochs, 
                              label_name, batch_size)
    
    #plot_the_loss_curve(epochs, mse)
    
    shutil.rmtree('models\\model1\\', ignore_errors=True, onerror=None)
    model.save('models\\model1\\')
#train()