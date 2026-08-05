import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

datafile = "MODULE 3/EDA/Ames_Housing_Data1.tsv"
df = pd.read_csv(datafile, sep='\t')

#print(df.info())

sns.histplot(df['Gr Liv Area'], kde = True)
#plt.show()

df = df.loc[df['Gr Liv Area'] <= 4000,:]
#print("Number of rows in the data :",df.shape[0])
#print("Number of columns in the data:",df.shape[1])
data = df.copy()

#print(df.head())

#dropping the unique values cuz we cant work with unique values in our model 
df.drop(["PID","Order"],axis = 1, inplace = True)

#log transforming and skewing variables
num_cols = df.select_dtypes('number').columns
#print(num_cols)

skew_limit = 0.75
skew_vals = df[num_cols].skew()
#print(skew_vals)

skew_cols = skew_vals[abs(skew_vals) > skew_limit].sort_values(ascending = False)
#print(skew_cols)

field = 'SalePrice'
fig, (ax_before,ax_after) = plt.subplots(1,2, figsize=(10, 5))
df[field].hist(ax = ax_before)
df[field].apply(np.log1p).hist(ax = ax_after)
ax_before.set(title = 'before np.log1p',ylabel = 'frequency',xlabel = 'value')
ax_after.set(title = 'after np.log1p',ylabel = 'frequency',xlabel = 'value')
fig.suptitle('field "{}"'.format(field));
#plt.show()

for col in skew_cols.index.values:
    if col == 'SalePrice':
        continue
    df[col] = df[col].apply(np.log1p)

#print(df.shape)

df = data
#print(data.isnull().sum().sort_values())
smaller_df= df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars','SalePrice']]

#print(smaller_df.describe().T)

#print(smaller_df.info())

smaller_df = smaller_df.fillna(0)
#print(smaller_df.info())

#pair plot of features
sns.pairplot(smaller_df, plot_kws = dict(alpha = .1, edgecolor = 'none'))
#plt.show()

X = smaller_df

y = smaller_df['SalePrice']
#print(X.info())

#basic feature engineering : adding polynomial and interaction terms 
X2 = X.copy()
X2['OQ2'] = X2['Overall Qual'] ** 2
X2['GLA2'] = X2['Gr Liv Area'] ** 2

X3 = X2.copy()
X3['OQ_x_YB'] = X3['Overall Qual'] * X3['Year Built']
X3['OQ_/_LA'] = X3['Overall Qual'] / X3['Lot Area']


#Categories and features derived from category aggregates 
print(data['House Style'].value_counts())
print(pd.get_dummies(df['House Style'], drop_first=True, dtype = int).head())

nbh_counts = df.Neighborhood.value_counts()
print(nbh_counts)

other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
print(other_nbhs)

X4 = X3.copy()
X4['Neighborhood'] = df['Neighborhood'].replace(other_nbhs, 'Other')

def add_deviation_feature(X,feature,category):
    category_gb = X.groupby(category)[feature]

    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())

    deviation_feature = (X[feature] - category_mean) / category_std 
    X[feature + '_Dev_' + category] = deviation_feature  

X5 = X4.copy()
X5['House Style'] = df['House Style']
add_deviation_feature(X5, 'Year Built', 'House Style')
add_deviation_feature(X5, 'Overall Qual', 'Neighborhood')   

print(X5)

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)

features = ['Lot Area', 'Overall Qual']
pf.fit(df[features])

print(pf.get_feature_names_out())

feat_array = pf.transform(df[features])
pd.DataFrame(feat_array, columns = pf.get_feature_names_out(input_features=features))