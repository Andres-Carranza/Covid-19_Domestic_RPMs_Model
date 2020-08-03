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

    max_vals = {
        'jan':1,'feb':1,'mar':1,'apr':1,'may':1,'jun':1,'jul':1,'aug':1,'sep':1,'oct':1,'nov':1,'dec':1,
        'leap-feb':1,'thanksgiving-nov':1,'thanksgiving-dec':1,
        'gulf-war':1, '9/2001':1, 'iraq-war':1,'sars-outbreak':1,'great-recession':1,
        'months-since-9/11':1,'months-since-covid-19':1,
        'unemployment-rate':14.7,'nonfarm-payroll':152463,
        'deaths':1955.033333,'new-infected':204180.4,'current-infected':3367528.167,
        'rpms':71230170772,
    }
    for col in data.columns:
        data[col] = data[col] / max_vals[col]

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

def train():
    
    training_data = pd.read_csv('data/training-data.csv')
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

    valid_data = pd.read_csv('data/validation-data.csv')
    valid_data = normalize_data(valid_data)

    test_features = {name:np.array(value) for name, value in valid_data.items()}
    test_label = np.array(test_features.pop('rpms')) # isolate the label
    print("\n Evaluate the  model against the test set:")
    model.evaluate(x = test_features, y = test_label, batch_size=batch_size)

    shutil.rmtree('model')
    model.save('model')
    mse.to_csv('model/mean_squared_error.csv', index = False)
    
    with open('model/summary.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))