import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
 
def download(url, filename):
    response =  requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
            
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/airlines_data.xlsx"
 
download(path, "MODULE 3/EDA/airlines_data.xlsx")
 
data = pd.read_excel("MODULE 3/EDA/airlines_data.xlsx")
 
#print(data.head())

#print(data.info())

#print(data.describe())

#print(data.isnull().sum())

data = data.fillna(method = "ffill")

#print(data["Airline"].unique().tolist())

data["Airline"] = np.where(data['Airline'] == "Vistara Premium economy", "Vistara", data['Airline'])
data["Airline"] = np.where(data['Airline'] == "Jet Airways Business", "Jet Arways", data['Airline'])
data["Airline"] = np.where(data['Airline'] == "Multiple carriers Premium economy", "Multiple carriers", data['Airline'])

#print(data['Airline'].unique().tolist())

#feature transformation
#ex1
data1 = pd.get_dummies(data=data,columns = ['Airline','Source','Destination'])
#print(data1.head())

#print(data.shape)
#print(data1.shape)

#print(data['Total_Stops'].value_counts)

#ex2
data1.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)
#print(data1.head())

duration = list(data1["Duration"])
for i in range (len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        elif "m" in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())
dur_hours = []
dur_minutes = []  
 
for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
     
 
data1['Duration_hours'] = dur_hours
data1['Duration_minutes'] =dur_minutes
data1.loc[:,'Duration_hours'] *= 60
data1['Duration_Total_mins']= data1['Duration_hours']+data1['Duration_minutes']
#print(data1.head())

data1['Dep_Hour'] = pd.to_datetime(data1['Dep_Time']).dt.hour
data1["Dep_Min"]= pd.to_datetime(data1['Dep_Time']).dt.minute

#ex3
data1['Arrival_Hour'] = pd.to_datetime(data1['Arrival_Time']).dt.hour
data1["Arrival_Min"]= pd.to_datetime(data1['Arrival_Time']).dt.minute

data1['dep_timezone'] = pd.cut(data1.Dep_Hour, [0,6,12,18,24], labels = ['Night','Morning','Afternoon','Evening'])
#print(data1['Dep_timezone'])

#ex4
data1['Arr_timezone'] = pd.cut(data1.Arrival_Hour, [0,6,12,18,24], labels = ['Night','Morning','Afternoon','Evening'])
#print(data1['Arr_timezone'])

#ex5
data1['Month']= pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.month
data1['Day']= pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.day
data1['Year']= pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.year
data1['day_of_week'] = pd.to_datetime(data1['Date_of_Journey']).dt.day_name()
#print(data1.head())

#feature selection
#print(data1.columns)
new_data = data1.loc[:,['Total_Stops', 'Airline_Air Asia',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Multiple carriers', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Source_Banglore',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi',
       'Duration_hours', 'Duration_minutes', 'Duration_Total_mins', 'Dep_Hour',
       'Dep_Min', 'dep_timezone', 'Price']]

plt.figure(figsize=(18,18))
sns.heatmap(new_data.corr(numeric_only=True),annot=True,cmap='RdYlGn')
#plt.show()

features = new_data.corr(numeric_only = True)['Price'][:-1].sort_values()
print(features)
features.plot(kind='bar',figsize=(18,18))
plt.show()