import skillsnetwork

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats 
from scipy.stats import chi2_contingency

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm 

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/insurance.csv'

data = pd.read_csv(url)
print(data.head())

print(data.info())#look at our data types
print(data.describe())#look at the statistical information about the numeric variables

#example 1 : approve or disapprove with statistical evidence that bmi of females i different from that of males
male = data.loc[data.sex == 'male']
female = data.loc[data.sex == 'female']

M_bmi = male.bmi
F_bmi = female.bmi

sns.distplot(F_bmi,color = 'pink',hist = False)
sns.distplot(M_bmi,color = 'blue',hist = False)
plt.show()

print(F_bmi.mean())
print(M_bmi.mean())

alpha = 0.05
t_value1,p_value1 = stats.ttest_ind(M_bmi,F_bmi)
print("t_value1 = ",t_value1, ", p_value1 = ", p_value1)

if p_value1 < alpha :
    print("Conclusion: since p_value {} is less than alpha {} ". format (p_value1,alpha))
    print("Reject the null hypothesis that there is no difference between bmi of females and bmi of males.")
else:
    print("Conclusion: since p_value {} is greater than alpha {} ". format (p_value1,alpha))
    print("Fail to reject the null hypothesis that there is a difference between bmi of females and bmi of males.")

#example 2 : prove or disapprove with statistical evidence that the medical claims made by the people who smoke are greater than those who don't
smoker = data.loc[data.smoker == 'yes']
non_smoker = data.loc[data.smoker == 'no']

smoker_charges = smoker.charges
sch = smoker_charges.mean()
print(sch)

non_smoker_charges = non_smoker.charges
nsch = non_smoker_charges.mean()
print(nsch)

sns.boxplot(x=data.charges,y=data.smoker,data=data).set(title='fig1 : smoker vs charges')
plt.show()

alpha = 0.05
t_value2,p_value2 = stats.ttest_ind(smoker_charges,non_smoker_charges)
print ("t_value2 = ",t_value2 ,", p_value2 = ",p_value2)

if p_value2 < alpha :
    print("Conclusion: since p_value {} is less than alpha {} ". format (p_value2,alpha))
    print("Reject the null hypothesis that the charges of smokers are less or equal to non smokers.")
else:
    print("Conclusion: since p_value {} is greater than alpha {} ". format (p_value2,alpha))
    print("Fail to reject the null hypothesis that the charges of smokers are less than non smokers")

#example 3 : using the statistical evidence we will compare the BMI of women with no children one child and two children
female_children = female.loc[female['children'] <= 2]
female_children.groupby('children')['bmi'].mean()
sns.boxplot(x='children',y='bmi',data=female_children).set(title = 'bmi distrubiton for females with no to 2 children')
plt.show()

formula = 'children ~ C(children)'
model = ols(formula,female_children).fit()
aov_table = anova_lm(model)
print(aov_table)

#example 4 : we will determine if the proportion of smokers is significantly different across the different regions
contingency = pd.crosstab(data.region,data.smoker)
print(contingency)

contingency.plot(kind ='bar').set(title = 'smokers across different regions')
plt.show()

chi2 , p_val , dof , exp_freq = chi2_contingency(contingency, correction = False)
print('chi-square statistic: {} , p_value: {} , degree of freedom: {} ,expected frequencies: {} '.format(chi2, p_val, dof, exp_freq))

if (p_val < 0.05):
    print('Reject the null hypothesis, that the smokers proportions are not significantly different across the different regions')
else:
    print('Accept the null hypothesis, that the smokers proportions are not significantly different across the different regions')
