import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
def plot(i):      
    data = pd.read_csv('data\\raw\\daily.csv')
    
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Covid-19')
    
    plt.xlabel('Date')
    plt.ylabel('#')

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for index in tck:
        tck_dates.append(data['Date'][index])
    plt.xticks(tck, tck_dates)
    
    data['Cases'] = data['Cases'].rolling(i).mean()
    data['Deaths'] = data['Deaths'].rolling(i).mean()
    
    data = data.replace(np.nan,0)
    data['Cases']/=max(data['Cases'])
    data['Deaths']/=max(data['Deaths'])
    data = data.drop(range(0,60))
    print(data)
    axes.plot(data['Date'], data['Cases'], label='Cases')
 
    axes.plot(data['Date'], data['Deaths'], label='Deaths')
 
    plt.legend(loc="upper left")
    
    
def plot_change(i): 
    data = pd.read_csv('data\\raw\\daily.csv')
    
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Covid-19')
    
    plt.xlabel('Date')
    plt.ylabel('#')
    date = data['Date'].drop(range(0,61)).values
    
    frq = 10
    step = max(int(len(date)/frq),1)
    tck = range(0,len(date), step)
    tck_dates = []
    for index in tck:
        tck_dates.append(date[index])
    plt.xticks(tck, tck_dates)
    
    change =  data['Cases'].rolling(i).mean()

    
    for index, row in enumerate(change):
        if index == 0:
            continue
        change.loc[index - 1] = change.loc[index] -change.loc[index - 1]

    change = change.drop(len(change)-1).rolling(14).mean().replace(np.nan,0)
    change/= max(change)    
    change = change.drop(range(0,60))

    axes.plot(date,change)
  
    change =  data['Deaths'].rolling(i).mean()
  
    for index, row in enumerate(change):
        if index == 0:
            continue
        change.loc[index - 1] = change.loc[index] -change.loc[index - 1]

    change = change.drop(len(change)-1).rolling(14).mean().replace(np.nan,0)
    change/= max(change)
    change = change.drop(range(0,60))
    axes.plot(date,change)
 
 
    plt.legend(loc="upper left")      
    
def cases(i):      
    data = pd.read_csv('data\\scenario1\\daily1.csv').drop(range(0,40)).reset_index()
    
    fig = plt.figure(figsize=(10,6))
    axes = fig.add_subplot(111)

    axes.set_title('Covid-19')
    
    plt.xlabel('Date')
    plt.ylabel('#')

    frq = 8
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for index in tck:
        tck_dates.append(data['Date'][index])
    plt.xticks(tck, tck_dates)
    

    axes.plot(data['Date'], data['Cases'], label='Cases')
    axes.plot(data['Date'], data['Cases'].rolling(i).mean())
    axes.plot(data['Date'], data['Deaths'], label='Deaths')
    axes.plot(data['Date'], data['Deaths'].rolling(i).mean())

    data = data.loc[40:100]
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(data.index.values.reshape(-1,1),data['Cases'].values.reshape(-1,1))  # perform linear regression
    reg = linear_regressor.predict(data.index.values.reshape(-1,1))
    
    print(linear_regressor.coef_)
    axes.plot(data['Date'], reg)
    
def percent(chng,i):
    data = pd.read_csv('data\\raw\\daily.csv')
    
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Covid-19')
    
    plt.xlabel('Date')
    plt.ylabel('#')
    date = data['Date'].drop(range(0,60 + chng))
    
    frq = 10
    step = max(int(len(date)/frq),1)
    tck = range(0,len(date), step)
    tck_dates = []
    for index in tck:
        tck_dates.append(date.values[index])
    plt.xticks(tck, tck_dates)
    
    cases =  data['Cases'].rolling(i).mean().drop(range(0,60))
    cases = cases.drop(range(len(cases), len(cases) - chng, -1))
    deaths =  data['Deaths'].rolling(i).mean().drop(range(0,60+chng))

    change = deaths.values/cases.values
    change = change
    

    print(date)
    print(change)
    axes.plot(change) 
 
    plt.legend(loc="upper left")    
    
i = 7
#plot(i)
#percent(14,i)
cases(i)
plt.show()