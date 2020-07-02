import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
def plot(i):      
    unp = pd.read_csv('data\\unemployement.csv')
    lab = pd.read_csv('data\\labor_force.csv')
   
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('econ')
    
    plt.xlabel('Date')
    plt.ylabel('#')

    data = pd.DataFrame()
    data['Date'] = unp['Date']

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for index in tck:
        tck_dates.append(data['Date'][index])
    plt.xticks(tck, tck_dates)
    
    data['lab'] = lab['lab'].rolling(i).mean()
    data['unp'] = unp['unp'].rolling(i).mean()
    
    data = data.drop(range(0,350))
    print(data)
 
    axes.plot(data['Date'], data['lab'], label='lab')
 
    plt.legend(loc="upper left")  
def plott(i):      
    unp = pd.read_csv('data\\unemployement.csv')
    lab = pd.read_csv('data\\labor_force.csv')
   
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('econ')
    
    plt.xlabel('Date')
    plt.ylabel('#')

    data = pd.DataFrame()
    data['Date'] = unp['Date']

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for index in tck:
        tck_dates.append(data['Date'][index])
    plt.xticks(tck, tck_dates)
    
    data['lab'] = lab['lab'].rolling(i).mean()
    data['unp'] = unp['unp'].rolling(i).mean()
    
    data = data.drop(range(0,350))
    print(data)
    axes.plot(data['Date'], data['unp'], label='unp')
 
    plt.legend(loc="upper left")      
    
i = 1
plot(i)
plott(i)
plt.show()