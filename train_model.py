import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import math
import shutil

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
 
def get_feature_layer(data):
    feature_columns = []
    for column in data.columns:
        if column == 'rpms':
            continue
        feature_columns.append(tf.feature_column.numeric_column(column))
    return tf.keras.layers.DenseFeatures(feature_columns)


def create_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    
    model.add(feature_layer)
    
    reg_rate = 0.00
    dropout_rate= 0.00

    model.add(tf.keras.layers.Dense(units=512,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    model.add(tf.keras.layers.Dense(units=512,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    model.add(tf.keras.layers.Dense(units=256,
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
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    model.add(tf.keras.layers.Dense(units=64,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(reg_rate) ))  
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,  
                                  name='Output'))                              
  
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

def train_model(model, dataset, epochs, label_name, batch_size=None):
    # Split the dataset into features and label.

    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    
    history = model.fit(x=features, y=label, batch_size=batch_size,epochs=epochs) 
    
    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch. 
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    
    return mse

def train(model_name):
    
    training_data = pd.read_csv('data/{}/training-data.csv'.format(model_name))
    print('Loaded data')
    
    training_data = normalize_data(training_data)
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
    label_name = "rpms"
    
    # Train the model on the normalized training set. We're passing the entire
    # normalized training set, but the model will only use the features
    # defined by the feature_layer.
    mse = train_model(model,training_data , epochs, 
                              label_name, batch_size)
    
    shutil.rmtree('models/{}'.format(model_name))
    model.save('models/{}'.format(model_name))
    mse.to_csv('models/{}/mean_squared_error.csv'.format(model_name), index = False)
    
    with open('models/{}/summary.txt'.format(model_name),'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

