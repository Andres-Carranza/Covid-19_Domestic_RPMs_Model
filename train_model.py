import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from matplotlib import pyplot as plt
import math
from datetime import datetime
import shutil

def apply_inverse(training_data,model):
    #Inverse squares of normalized data
    training_data['MSC'] = 1 / training_data['MSC'] **.75
    training_data['Months_After_9/11'] = 1 / training_data['Months_After_9/11'] **2
    training_data = training_data.replace(math.inf,0)

    return training_data

def get_normalized_data(model = 1):
    training_data = pd.read_csv('data\\training_data.csv')
    

    training_data = apply_inverse(training_data,model)

    training_data['Months_After_9/11'] = training_data['Months_After_9/11'] / max(training_data['Months_After_9/11'])

 
    training_data['MSC'] = training_data['MSC'] / max(training_data['MSC'])

    training_data['Unemployement'] = training_data['Unemployement'] / max(training_data['Unemployement'])
    
    training_data['Labor_Force'] = training_data['Labor_Force'] / max(training_data['Labor_Force'])

    training_data['Avg'] = training_data['Avg']/ max(training_data['Avg'])
    training_data['Avgd'] = training_data['Avgd']/ max(training_data['Avgd'])

    training_data['RPMs'] = training_data['RPMs'] / max(training_data['RPMs'])

    return training_data

def get_max_vals(model =1):
    training_data = pd.read_csv('data\\training_data.csv')

    training_data = apply_inverse(training_data,model)

    return { 
                'Months_After_9/11': max(training_data['Months_After_9/11']),
                'Unemployement': max(training_data['Unemployement']),
                'Labor_Force': max(training_data['Labor_Force']),
                'RPMs': max(training_data['RPMs']),
                'Avg':max(training_data['Avg']),
                'Avgd':max(training_data['Avgd']),
                'MSC':max(training_data['MSC'])}    
def get_feature_layer(data):
    feature_columns = []
    for column in data.columns:
        if column in['RPMs','Date','Labor_Force'] :
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
    dropout_rate= 0.005
    model.add(tf.keras.layers.Dense(units=1000,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dense(units=500,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dense(units=500,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))    
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(units=300,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))    
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(units=300,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(units=300,
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
    del dataset['Labor_Force']
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
    
    shutil.rmtree('models\\model2\\', ignore_errors=True, onerror=None)
    model.save('models\\model2\\')
#train()