#ex1
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

import requests

#Data Acquisition + Data Exploration (EDA)
def download(url, filename):
    res = requests.get(url)

    if res.status_code == 200 :
        with open(filename, "wb") as f:
            f.write(res.content)

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv"
download(path, "MODULE 2/CLEANING DATA/Ames_Housing_Data1.tsv")
housing = pd.read_csv("MODULE 2/CLEANING DATA/Ames_Housing_Data1.tsv", sep="\t")

#print(housing.head(5))

#print(housing.info())

#print(housing["SalePrice"].describe())

#print(housing["Sale Condition"].value_counts())


#Correlation
house_num = housing.select_dtypes(include = ['float64','int64'])
#print(house_num)
house_num_corr = house_num.corr()['SalePrice'][:-1]
top_features = house_num_corr[abs(house_num_corr) > 0.5].sort_values(ascending=False) 
#print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

"""
for i in range(0, len(house_num.columns), 5):
    sns.pairplot(data=house_num,
                x_vars=house_num.columns[i:i+5],
                y_vars=['SalePrice'],)
    #plt.show()
"""

#Log transformation
sp_untransformed = sns.histplot(
    housing['SalePrice'],
    kde=True
)

#plt.show()

#print("Skewness: %f" % housing['SalePrice'].skew())

log_transformed = np.log(housing['SalePrice'])

sp_transformed = sns.histplot(log_transformed,
                              kde = True)

#print("Skewness: %f" % (log_transformed).skew())
#plt.show()

#ex2
la_untransformed = sns.histplot(
    housing["Lot Area"],
    kde = True
)

#print("skewness : %f" % housing["Lot Area"].skew())

la_log = np.log(housing["Lot Area"])

#print("skewness : %f" % la_log.skew())

#plt.show()

duplicate = housing[housing.duplicated(['PID'])]
print(duplicate)

dup_removed = housing.drop_duplicates()
print(dup_removed)

print(housing.index.is_unique)

#ex3
removed_sub = housing.drop_duplicates(subset = ["Order"])
print(removed_sub)

total = housing.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

#plt.show()

housing.dropna(subset=["Lot Frontage"])
housing.drop("Lot Frontage", axis=1)

median = housing["Lot Frontage"].median()
print("median",median)

housing["Lot Frontage"].fillna(median, inplace = True)
print(housing.tail())

#ex4
mean = housing["Mas Vnr Area"].mean()
housing["Mas Vnr Area"].fillna(mean, inplace = True) 

norm_data = MinMaxScaler().fit_transform(house_num)
print("norm data",norm_data)

scaled_data = StandardScaler().fit_transform(house_num)
print("scaled data",scaled_data)

#ex5
scaled_sprice = StandardScaler().fit_transform(
    housing['SalePrice'].to_numpy().reshape(-1, 1)
)
print("scaled_sprice",scaled_sprice)

sns.boxplot(x=housing['Lot Area'])

sns.boxplot(x=housing['SalePrice'])

price_area = housing.plot.scatter(x='Gr Liv Area',
                      y='SalePrice')

housing.sort_values(by = 'Gr Liv Area', ascending = False)[:2]

outliers_dropped = housing.drop(housing.index[[1499,2181]])

new_plot = outliers_dropped.plot.scatter(x='Gr Liv Area',
                                         y='SalePrice')

#ex6
sns.boxplot(x=housing['Lot Area'])
price_lot = housing.plot.scatter(x='Lot Area', y='SalePrice')   
housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])
housing[['Lot Area','Lot_Area_Stats']].describe().round(3)
housing.sort_values(by = 'Lot Area', ascending = False)[:1]
lot_area_rem = housing.drop(housing.index[[957]])

housing['LQFSF_Stats'] = stats.zscore(housing['Low Qual Fin SF'])
housing[['Low Qual Fin SF','LQFSF_Stats']].describe().round(3)