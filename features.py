import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_rpms():
    data = pd.read_csv('data\\domestic_rpms.csv')
    
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Monthly Domestic RPMs Time Series')
    
    plt.xlabel('Date')
    plt.ylabel('RPMs (billions)')
    
    low = min(data['Domestic LCC RPMs'])/1000000000
    high = max(data['Total Domestic RPMs'])/1000000000
    rng = high-low
    axes.set_ylim(int(low-rng*.1), int(high+rng*.1))

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(data['Date'][i])
    plt.xticks(tck, tck_dates)
    
    axes.plot(data['Date'], data['Total Domestic RPMs']/1000000000, label='Total Domestic RPMs')
    axes.plot(data['Date'], data['Domestic LCC RPMs']/1000000000, label='Monthly LCC RPMs')
    axes.plot(data['Date'], data['Total Domestic RPMs'].rolling(12).mean()/1000000000, label='12 Month Rolling mean')
    plt.legend(loc="upper left")
