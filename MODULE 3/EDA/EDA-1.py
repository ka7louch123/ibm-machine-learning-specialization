import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
import requests 

#QUESTION 1
def download(url,filename):
    res = requests.get(url)

    if res.status_code == 200:
        with open(filename,'wb') as f :
            f.write(res.content)

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv"
download(path, "iris_data.csv") 
data = pd.read_csv("iris_data.csv")

print(data.head())
print(data.shape[0])
print(data.columns.tolist())
print(data.dtypes)

#QUESTION 2
data['species'] = data.species.str.replace('Iris-', '')

print(data.head())

#QUESTION 3
#print(data.species.value_counts())

#print(data.describe())

stats_df = data.describe()
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']

out_fields = ['mean','25%','50%','75%', 'range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%': 'median'}, inplace=True)

print(stats_df)

#QUESTION 4
print(data.groupby('species').mean())
print(data.groupby('species').median())

#another solution
data.groupby('species').agg(['mean', 'median'])  
data.groupby('species').agg([np.mean, np.median])

from pprint import pprint

agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
agg_dict['petal_length'] = 'max'
pprint(agg_dict)
data.groupby('species').agg(agg_dict)

#QUESTION 5
ax = plt.axes()

ax.scatter(data.sepal_length, data.sepal_width)
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width');
#plt.show()

#QUESTION 6
ax = plt.axes()

# Using Matplotlib's plotting functionality
ax.hist(data.petal_length, bins=25);
ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths');
#plt.show()

#alternative using panda plotting functionality
ax = data.petal_length.plot.hist(bins=25)
ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths');
#plt.show()

#QUESTION 7
ax = data.plot.hist(bins=25, alpha=0.5)
ax.set_xlabel('Size (cm)');

axList = data.hist(bins=25)

for ax in axList.flatten():
        ax.set_xlabel('Size (cm)')
        ax.set_ylabel('Frequency')
#plt.show()

#QUESTION 8
data.boxplot(by='species');
#plt.show()

#QUESTION 9
plot_data = (data
             .set_index('species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

print(plot_data.head())

sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

f = plt.figure(figsize=(6,4))
sns.boxplot(x='measurement', y='size', 
            hue='species', data=plot_data);

#QUESTION 10
sns.set_context('talk')
sns.pairplot(data, hue='species');

plt.show()