import pandas as pd
data=pd.read_csv('StudentsPerformance.csv')


print(" ")
print("ex1:")
print(data.head)
print(data.head())
print (data.info())
print(data.describe())
print(data.isnull().sum())

print("ex2:")
numerice=data.select_dtypes(include=['number']).columns
categorical=data.select_dtypes(include=['category']).columns
print(numerice)
print(categorical)

print("ex3:")

categorice = data.select_dtypes(include=['object', 'string']).columns

for col in numerice:
    mediana = data[col].median()
    data[col] = data[col].fillna(mediana)


for col in categorice:
    data[col] = data[col].fillna("Unknown")


print("--- Verificare finală valori lipsă ---")
print(data.isnull().sum())
