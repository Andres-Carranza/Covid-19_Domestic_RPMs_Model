import pandas as pd
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

def load_domestic_data():
    yearly_data_path = 'data\\yearly_data\\{}.csv'
    
    lcc_ids = pd.read_csv('data\\lcc_ids.csv')['id'].to_list()
    
    start_year = 1990
    end_year = 2020
    start_date = start_year * 12 + 1
    end_date = end_year * 12 + 3
    
    dates = []
    for date in range(start_date, end_date + 1):
        year = date // 12
        month = date % 12
        
        if month == 0:
            month = 12
            year-=1
        dates.append('{}/{}'.format(month,year))
    
    columns = ['Date','Domestic LCC RPMs','Total Domestic RPMs']
    
    rpms_data = pd.DataFrame(columns=columns, data=zip(dates,[0] * len(dates),[0] * len(dates)))
    
    
    for year in tqdm(range(start_year, end_year + 1)):
        yearly_data = pd.read_csv(yearly_data_path.format(year))
        
        for index, row in yearly_data.iterrows():
    
            month = int(row['MONTH'])     
        
            rel_date = year * 12 + month - start_date
    
            try:
                rpms_data.at[rel_date,'Total Domestic RPMs']+= row['PASSENGERS'] * row['DISTANCE']
    
            
                if row['AIRLINE_ID'] in lcc_ids:
                    rpms_data.at[rel_date,'Domestic LCC RPMs']+= row['PASSENGERS'] * row['DISTANCE']
            except ValueError:
                print('ValueError: most likely NaN')
            except:
                print('Unknown exception caught')
    print(rpms_data)
    
    rpms_data.to_csv('domestic_rpms.csv', index = False)

def load_unemployement():
    db = pd.read_csv('data\\raw_unemployement.csv')
        
    ndb = pd.DataFrame(columns=['Date','Unemployement'])
    
    for index, row in db.iterrows():
        for month, value in enumerate(row[1:]):
            date = '{}/{}'.format(month+1, int(row[0]))
            ndb.loc[index*12+month] = [date,value]
    ndb.to_csv('unemployement.csv',index = false)

def load_labor_force():
    db = pd.read_csv('data\\raw_force.csv')
        
    ndb = pd.DataFrame(columns=['Date','Labor Force'])
    
    for index, row in db.iterrows():
        for month, value in enumerate(row[1:]):
            date = '{}/{}'.format(month+1, int(row[0]))
            ndb.loc[index*12+month] = [date,value]
    ndb.to_csv('labor_force.csv',index = False)
            
def load_fuel_prices():
    raw = pd.read_csv('data\\jetraw.csv')
    
    start_year = 1990
    end_year = 2020
    start_date = start_year * 12 + 1
    end_date = end_year * 12 + 3
    
    dates = []
    for date in range(start_date, end_date + 1):
        year = date // 12
        month = date % 12
        
        if month == 0:
            month = 12
            year-=1
        dates.append('{}/{}'.format(month,year))
    
    columns = ['Date','G','C','CPG']
    
    fuel_data = pd.DataFrame(columns=columns, data=zip(dates,[0] * len(dates),[0] * len(dates),[0] * len(dates)))

    for index, row in tqdm(raw.iterrows()):
        year = int(row['YEAR'])     
        month = int(row['MONTH'])     
        
        rel_date = year * 12 + month - start_date
    
        try:                
            fuel_data.at[rel_date,'C']+= row['TDOMT_COST']
            fuel_data.at[rel_date,'G']+= row['TDOMT_GALLONS']
        except ValueError:
            continue
        except:
            print('Unknown exception caught')
            
    fuel_data['CPG'] = fuel_data['C'] / fuel_data['G']
    
    del fuel_data['C']
    del fuel_data['G']

    print(fuel_data)
    
    fuel_data.to_csv('domestic_cpg.csv', index = False)
    
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

def plot_lcc_market_share():
    data = pd.read_csv('data\\domestic_rpms.csv')
    
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('LCC Domestic Market Share Time Series')
    
    plt.xlabel('Date')
    plt.ylabel('LCC Market Share')

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(data['Date'][i])
    plt.xticks(tck, tck_dates)
    
    axes.plot(data['Date'], data['Domestic LCC RPMs']/data['Total Domestic RPMs'])
    
def plot_unemployement():
    data = pd.read_csv('data\\unemployement.csv')
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Unemployement Rate Time Series')
    
    plt.xlabel('Date')
    plt.ylabel('Unemployement Rate')

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(data['Date'][i])
    plt.xticks(tck, tck_dates)
    
    axes.plot(data['Date'], data['Unemployement'])
    
def plot_labor_force():
    data = pd.read_csv('data\\labor_force.csv')
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Labor Force Time Series')
    
    plt.xlabel('Date')
    plt.ylabel('Labor Force (thousands)')

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(data['Date'][i])
    plt.xticks(tck, tck_dates)
    
    axes.plot(data['Date'], data['Labor Force'])
    
def plot_cpg():
    data = pd.read_csv('data\\domestic_cpg.csv')
    fig = plt.figure(figsize=(15,2))
    axes = fig.add_subplot(111)

    axes.set_title('Jet Fuel Cost Per Gallon Time Series')
    
    plt.xlabel('Date')
    plt.ylabel('CPG')

    frq = 10
    step = max(int(len(data['Date'])/frq),1)
    tck = range(0,len(data['Date']), step)
    tck_dates = []
    for i in tck:
        tck_dates.append(data['Date'][i])
    plt.xticks(tck, tck_dates)
    
    axes.plot(data['Date'], data['CPG'])
plot_rpms()
plot_lcc_market_share()
#plot_unemployement()
#plot_cpg()
#plot_labor_force()

#load_domestic_data()
plt.show()
