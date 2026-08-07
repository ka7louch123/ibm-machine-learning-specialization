import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
sns.set()

# TASK 1: DATASET SUMMARY
data = pd.read_csv('laptops.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isna().sum())

"""
Task 1: Dataset Summary

The laptops dataset contains numerical and categorical variables
describing laptop specifications, including brand, model, RAM,
storage, screen size, GPU, touchscreen support, condition, and
final price.

The Final Price column is the main variable examined in this
analysis. The head(), info(), describe(), and isna().sum()
functions were used to inspect the dataset structure, data types,
summary statistics, and missing values.
"""

# TASK 2: DATA EXPLORATION PLAN
"""
Task 2: Data Exploration Plan
The objective of this analysis is to understand the characteristics
that influence laptop prices.
The analysis includes:
1. Inspecting the dataset and identifying missing values.
2. Cleaning missing or incomplete data.
3. Creating useful features from existing variables.
4. Examining the distribution and skewness of numerical variables.
5. Applying transformations to highly skewed features.
6. Studying correlations between numerical variables.
7. Comparing laptop prices across different groups.
8. Formulating and testing three statistical hypotheses.
"""

# TASK 3&4: EXPLORATORY DATA ANALYSIS & DATA CLEANING AND FEATURE ENGINEERING
data['GPU'] = np.where(data['GPU'].isna(), 'Integrated', data['GPU'])

data['Screen'] = (data['Screen'].fillna(data.groupby(['Brand','Model'])['Screen'].transform('median'))
                  .fillna(data.groupby('Brand')['Screen'].transform('median'))
                  .fillna(data['Screen'].median()))

is_missing = data['Storage type'].isna()
cond_eMMC = (data['Storage type'].isna() & (0 < data['Storage']) & (data['Storage'] <= 64))
cond_SSD = (data['Storage type'].isna() & ((data['Storage'] >= 128) | (data['Storage'] == 0))) 
data['Storage type']= np.select([cond_eMMC,cond_SSD],['eMMC','SSD'],default = data['Storage type'])

print(data.info())
print(data.isna().sum())

data['Is_refurbished'] = np.where(data['Status'] == 'Refurbished',1,0)
data['Screen_touch'] = np.where(data['Touch'] == 'Yes',1,0)

sns.histplot(data['Final Price'], kde =True)
plt.title('Distribution of laptop prices (£)')
plt.xlabel('Final price (£)')
plt.ylabel('Price distribution')

field = 'Final Price'
fig, (ax_before,ax_after) = plt.subplots(1,2, figsize=(10,5))
data[field].hist(ax = ax_before)
data[field].apply(np.log1p).hist(ax = ax_after)
ax_before.set(title = 'final price before np1log',ylabel = 'frequency',xlabel = 'value')
ax_after.set(title = 'final price after np1log',ylabel = 'frequency',xlabel = 'value')
fig.suptitle('field "{}"'.format(field))
#plt.show()

num_cols = data.select_dtypes(include="number").columns

skew_limit = 0.75
skew_vals = data[num_cols].skew()

skew_cols = skew_vals[abs(skew_vals) > skew_limit].sort_values(ascending=False)

data["Has_Dedicated_GPU"] = np.where(data["GPU"] != "Integrated",1,0)

binary_cols = ["Screen_touch","Is_refurbished","Has_Dedicated_GPU"]

for col in skew_cols.index:
    if col == "Final Price" or col in binary_cols:
        continue
    data["Log_" + col] = np.log1p(data[col])
data["Log_Final_Price"] = np.log1p(data["Final Price"])

text_categorical_cols = ["Brand", "Storage type"]

data_encoded = pd.get_dummies(data,columns=text_categorical_cols,drop_first=True,dtype=int)

updated_num_cols = data.select_dtypes(include="number").columns

plt.figure(figsize=(12,8))
sns.heatmap(data[updated_num_cols].corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Numerical Feature Correlation Matrix")

plt.figure(figsize=(8,5))
sns.boxplot(x="Has_Dedicated_GPU",y="Final Price",data=data)
plt.title("Final Price by GPU Type")
plt.xlabel("Has Dedicated GPU")
plt.ylabel("Final Price (£)")
plt.show()

data = data.drop(columns=["GPU", "Touch", "Status"])

"""
Task 3: Exploratory Data Analysis
Exploratory Data Analysis was performed to examine the distribution
and relationships between the dataset variables.
A histogram with a KDE curve showed that Final Price is positively
skewed. A log1p transformation produced a more symmetric price
distribution.
Skewness was calculated for all numerical variables. Variables
with absolute skewness greater than 0.75 were identified, and
separate log-transformed features were created while preserving
the original columns.
A correlation heatmap was used to examine relationships between
numerical variables. A boxplot was also used to compare laptop
prices between devices with integrated and dedicated GPUs.
"""

"""
Task 4: Data Cleaning and Feature Engineering
Missing GPU values were replaced with Integrated. Missing Screen
values were filled using the median screen size for laptops with
the same Brand and Model, followed by the Brand median and the
overall median.
Missing Storage type values were inferred from storage capacity.
Storage values up to 64 GB were classified as eMMC, while values
of 128 GB or more were classified as SSD.
Three binary features were created:
- Is_refurbished indicates whether a laptop is refurbished.
- Screen_touch indicates whether a laptop has a touchscreen.
- Has_Dedicated_GPU indicates whether a laptop has dedicated
  graphics.
"""

# TASK 5: KEY FINDINGS AND INSIGHTS
"""
Task 5: Key Findings and Insights
The Final Price variable is positively skewed, with a smaller
number of expensive laptops producing a long right tail.
The log1p transformation produced a more symmetric price
distribution.
Missing values were handled without deleting observations.
Feature engineering created useful variables such as
Has_Dedicated_GPU, Screen_touch, and Is_refurbished.
The correlation heatmap shows relationships between several
numerical features and laptop price.
The GPU boxplot suggests that laptops with dedicated GPUs
generally have higher prices than laptops with integrated GPUs.
"""

# TASKS 6&7: HYPOTHESIS FORMULATION & HYPOTHESIS TESTING
"""
Hypothesis 1:
H0: Laptops with dedicated and integrated GPUs have the same
    average price.
H1: Laptops with dedicated and integrated GPUs have different
    average prices.
"""
alpha = 0.05
dedicated_GPU_laptops = data.loc[data['Has_Dedicated_GPU'] == 1,'Final Price']
integrated_GPU_laptops = data.loc[data['Has_Dedicated_GPU'] == 0,'Final Price']

sns.distplot(dedicated_GPU_laptops,label = 'dedicated gpu',color = 'green',hist = False)
sns.distplot(integrated_GPU_laptops,label = 'integrated gpu',color = 'blue',hist = False)
plt.title('Price Distribution by GPU')
plt.show()

print("Average price (Dedicated GPU): £", dedicated_GPU_laptops.mean())
print("Average price (Integrated GPU): £", integrated_GPU_laptops.mean())

t_value1,p_value1 = stats.ttest_ind(dedicated_GPU_laptops,integrated_GPU_laptops,equal_var = False)
print('t_value1 : ',t_value1,'p_value1 : ',p_value1)

if p_value1 < alpha :
    print("since the p_value1 {} is less than alpha {}".format(p_value1,alpha))
    print("Reject the null hypothesis")
    print("There is sufficient statistical evidence to conclude that laptops with dedicated GPUs have significantly different prices.")
else:
    print("Conclusion: since p_value {} is greater than alpha {} ". format (p_value1,alpha))
    print("There is insufficient statistical evidence to conclude that dedicated GPU laptops have different prices")

"""
Hypothesis 2:
H0: Laptops with 16 GB RAM or less and laptops with more than
    16 GB RAM have the same average price.
H1: The two RAM groups have different average prices.
"""
normal_ram_laptops = data.loc[(0 <= data['RAM']) & (data['RAM'] <= 16),'Final Price']
high_ram_laptops = data.loc[16 < data['RAM'],'Final Price']

sns.distplot(normal_ram_laptops,label = '8gb & 16gb ram laptops',color = 'black',hist = False)
sns.distplot(high_ram_laptops,label = 'above 16gb ram laptops',color = 'yellow',hist = False)
plt.title("Price Distribution by RAM")
plt.show()

print("Average price (8gb & 16gb ram laptops): £", normal_ram_laptops.mean())
print("Average price (above 16gb ram laptops): £", high_ram_laptops.mean())

t_value2,p_value2 = stats.ttest_ind(normal_ram_laptops,high_ram_laptops,equal_var = False)
print('t_value2 : ',t_value2,'p_value2 : ',p_value2)

if p_value2 < alpha :
    print("since the p_value2 {} is less than alpha {}".format(p_value2,alpha))
    print("Reject the null hypothesis")
    print("There is sufficient statistical evidence to conclude that laptops with more RAM have different prices")   
else:
    print("Conclusion: since p_value {} is greater than alpha {} ". format (p_value2,alpha))
    print("There is insufficient statistical evidence to conclude that laptops with more RAM have different prices")

"""
Hypothesis 3:
H0: Refurbished and new laptops have the same average price.
H1: Refurbished and new laptops have different average prices.
"""
refurbished_laptops = data.loc[data["Is_refurbished"] == 1,"Final Price"]

new_laptops = data.loc[data["Is_refurbished"] == 0,"Final Price"]

plt.figure(figsize=(8, 5))
sns.distplot(refurbished_laptops,label="Refurbished laptops",color="red",hist=False)
sns.distplot(new_laptops,label="New laptops",color="pink",hist=False)
plt.title("Price Distribution by Status")
plt.show()

print("Average price (refurbished laptops): £",refurbished_laptops.mean())
print("Average price (new laptops): £",new_laptops.mean())

t_value3, p_value3 = stats.ttest_ind(refurbished_laptops,new_laptops,equal_var=False)
print("t_value3:", t_value3)
print("p_value3:", p_value3)

if p_value3 < alpha:
    print("Since p_value3 {} is less than alpha {}.".format(p_value3, alpha))
    print("Reject the null hypothesis.")

    if refurbished_laptops.mean() < new_laptops.mean():
        print("There is sufficient statistical evidence to conclude that refurbished laptops are cheaper on average")
    else:
        print("There is a significant difference but refurbished laptops are not cheaper on average")
else:
    print("Since p_value3 {} is greater than or equal to alpha {}.".format(p_value3, alpha))
    print("Fail to reject the null hypothesis.")
    print("There is insufficient statistical evidence to conclude that refurbished laptops are cheaper")

# TASK 8: CONCLUSION AND NEXT STEPS
"""
Task 8: Conclusion and Next Steps
The laptop dataset was successfully cleaned and explored using
missing-value treatment, feature engineering, categorical encoding,
log transformations, and visual analysis.
Missing GPU, Screen, and Storage type values were handled without
deleting observations. New variables were created to represent
refurbishment status, touchscreen support, and dedicated GPU
availability.
Three hypotheses were tested using independent Welch's t-tests at
a significance level of 0.05. The p-values were used to determine
whether the null hypotheses should be rejected, while the group
averages were used to interpret the direction of each difference.
As a next step, the encoded dataset can be divided into training
and testing sets and used to train laptop price prediction models.
Possible models include Linear Regression, Decision Tree, Random
Forest, and Gradient Boosting. Model performance can be evaluated
using MAE, RMSE, and R-squared.
"""