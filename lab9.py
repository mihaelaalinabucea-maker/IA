import pandas as pd
from pandas import set_option

data=pd.read_csv('data.csv')
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(data)

#2
jucatori_peste_40ani=data[data['Age']>40].head(10)
print(jucatori_peste_40ani)

#3
jucatori_over_age=data[(data['Overall']>=85) & (data['Age']<25)]
print(jucatori_over_age)

#4
data_sm=data.sort_values("Skill Moves",ascending=False).head(3)
print(data_sm)

#5
jucatori_2021=data[data['Contract Valid Until']=='2021'].head()
print(jucatori_2021)






