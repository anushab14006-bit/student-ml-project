import pandas as pd


df=pd.read_csv('student_data.csv')
X = df[['G1','G2']]
y = df['G3']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()



model.fit(X_train,y_train)

predictions=model.predict(X_test)
print(predictions[:5])

from sklearn.metrics import r2_score

print(r2_score(y_test, predictions))



#print(df)
#print(df.head(5))
#print(df.tail(5))

#READ HEADER

#print(df.columns)
#print(df['age'])
#print(df['age'][0:5])
#print(df[['age','sex','G1']])

# READ EACH ROW

#print(df.iloc[1:4])
#print(df.iloc[2,0])
#for index,row in df.iterrows():
   # print(index,row['sex'])
#print(df.loc[df['sex']=="F"])

#SIRTING/DESCRINDING

#print(df.describe())
#print(df.sort_values('age',ascending=False))
#print(df.sort_values['age','absence'],ascending=[1,0])its wrong
#print(df.sort_values(['age', 'absences'], ascending=[1, 0]))    both are same print(df.sort_values(by=['age', 'absences'], ascending=[True, False]))

#MAKING CHANGES TO THE DATA

#df['total']=df['G1']+df['G2']+df['G3']
#print(df.head())
#df = df.drop(columns=['total'])
#print(df.head())
#cols=list(df.columns)
#df=df[cols[0:10]+[cols[-1]]+cols[10:34]]
#print(df.head(5))

#SAVING OUR DATA

#df.to_csv('modified.csv',index=False)
#df.to_excel('modified.xlsx')  in terminal we should use pip install openpyxl
#df.to_csv('modified.txt',sep='\t')


#FILTERING DATA

#print(df)
#print(df.loc[(df['G1']==5) & (df['sex']=='F')])
#print(df.loc[(df['G1']>5) & (df['sex']=='F') | (df['address']=='U')])

import re

#df.loc[df['G1'].str.contains(5,regex=True)]
#print(df.loc[df['G1'].astype(str).str.contains('5')])

#CONDITIONAL CHANGES

#df.loc[df['sex'] == 'F', 'sex'] = 'U'
#print(df.loc[df['sex'] == 'U'])