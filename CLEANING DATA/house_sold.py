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
download(path, "Ames_Housing_Data1.tsv")
housing = pd.read_csv("Ames_Housing_Data1.tsv", sep="\t")

#print(housing.head(5))

#print(housing.info())

#print(housing["SalePrice"].describe())

#print(housing["Sale Condition"].value_counts())


#Correlation
house_num = housing.select_dtypes(include = ['float64','int64'])
print(house_num)
house_num_corr = house_num.corr()['SalePrice'][:-1]
top_features = house_num_corr[abs(house_num_corr) > 0.5].sort_values(ascending=False) 
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

for i in range(0, len(house_num.columns), 5):
    sns.pairplot(data=house_num,
                x_vars=house_num.columns[i:i+5],
                y_vars=['SalePrice'],)
    plt.show()


#Log transformation
sp_untransformed = sns.histplot(
    housing['SalePrice'],
    kde=True
)

#plt.show()

print("Skewness: %f" % housing['SalePrice'].skew())

log_transformed = np.log(housing['SalePrice'])

sp_transformed = sns.histplot(log_transformed,
                              kde = True)

print("Skewness: %f" % (log_transformed).skew())
#plt.show()