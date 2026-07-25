import pandas as pd 
import numpy as np
import matplotlib .pylab as plt
import seaborn as sns

df = pd.read_csv("EX/Messy_Employee_dataset.csv")

print("head",df.head())
print("shape",df.shape)
print("info", df.info())
print("description", df.describe())
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#cleaning the phones
df['Phone'] = df['Phone'].abs().astype(str).str.zfill(10).apply(
    lambda x: f"({x[:3]}) {x[3:6]}-{x[6:]}"
)

print(df['Phone'].head())

#splitting one in two columns
df[["Department" , "Region"]] = df['Department_Region'].str.split('-',expand = True)
df.drop(columns=['Department_Region'], inplace=True)

print(df.head())

#fixing age NaN value
print(df["Age"].isnull().sum())
df['Age'] = df['Age'].fillna(df.groupby('Department')['Age'].transform('median'))

print(df.head())

#fixing the salary
print(df["Salary"].isnull().sum())
df['Salary'] = df['Salary'].fillna(df.groupby(['Department','Performance_Score'])['Salary'].transform('median'))

print(df.head())

#fix join date type
df['Join_Date'] = pd.to_datetime(df['Join_Date'])

#Plot 1: Check for remaining missing values
sns.heatmap(df.isnull(), cbar=False, cmap="mako", ax=axes[0, 0])
axes[0, 0].set_title("1. Missing Values Check (Solid Color = Zero Missing Data)", fontsize=12)

#Plot 2: Salary Distribution
sns.histplot(df['Salary'], kde=True, ax=axes[0, 1], color="teal", bins=20)
axes[0, 1].set_title("2. Cleaned Salary Distribution", fontsize=12)

#Plot 3: Department Breakdown
department_counts = df['Department'].value_counts()
sns.barplot(x=department_counts.index, y=department_counts.values, ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("3. Employees per Department", fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=30)

#Plot 4: Salary vs Department Boxplot
sns.boxplot(x='Department', y='Salary', data=df, ax=axes[1, 1], palette="Blues")
axes[1, 1].set_title("4. Salary Spread across Departments", fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()
df.to_csv("Cleaned_Employee_dataset.csv", index=False)