import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('EXS/ex2/healthcare_messy_data.csv')

#print('head',df.head())
#print('info',df.info())
#print('describe',df.describe())
#print('shape',df.shape)

#cleaning the phones
df['Phone Number'] = np.where(df['Phone Number'].isna(), "000-000-000",df['Phone Number'])
#print(df['Phone Number'])

#cleaning emails
df['Email'] = np.where(df['Email'].isna(), "no_email@example.com",df['Email'])
#print(df['Email'])

#cleaning cholesterol
avg_cholesterol = df['Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].fillna(avg_cholesterol)
#print(df['Cholesterol'])

#cleaning the blood pressure
df[['Systolic','Dialostic']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
df['Systolic'] = df['Systolic'].fillna(df['Systolic'].mean())
df['Dialostic'] = df['Dialostic'].fillna(df['Dialostic'].mean())
df['Blood Pressure Cleaned'] = (df['Systolic'].round().astype(int).astype(str) + '/' + df['Dialostic'].round().astype(int).astype(str))
#print(df['Blood Pressure Cleaned'])

#cleaning the visit date
df['Visit Date'] = pd.to_datetime(df['Visit Date'], format = 'mixed')
df['Visit Year'] = df['Visit Date'].dt.year
df['Visit Month'] = df['Visit Date'].dt.month
df['Visit Day'] = df['Visit Date'].dt.day
df['Visit DayOfTheweek'] = df['Visit Date'].dt.dayofweek
df['Visit Date Clean'] = df['Visit Date'].dt.strftime('%Y-%m-%d')
#print(df['Visit Date Clean'])

#cleaning the medecine issue
df['Medication'] = df['Medication'].fillna("UNKNOWN")
#print(df['Medication'])

#cleaning the age issue
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
avg_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(avg_age).astype(int)
#print(df['Age'])

print(df.head())

sns.histplot(df['Age'], kde = True, bins = 15, color = 'skyblue')
plt.title('Age distribution of patients')
plt.xlabel('Age')
plt.ylabel('Count')
#plt.show()

df['High Cardio Risk'] = np.where((df['Cholesterol'] >= 200) & (df['Systolic'] >= 130),1,0)
print(df['High Cardio Risk'].value_counts)

plt.figure(figsize = (7,5))
sns.countplot(data = df, x = 'High Cardio Risk')
plt.title('Patient Count: High Cardiovascular Risk Flag', fontsize = 13)
plt.xticks(ticks = [0,1], labels = ['Standart Risk (0)','High Risk (1)'])
plt.xlabel('CradioVascular Risk Category')
plt.ylabel('Number of patients')
plt.show()

df.to_csv("EXS/ex2/Cleaned_Employee_dataset.csv", index=False)