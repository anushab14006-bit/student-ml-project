import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('student_data.csv')

# -------- Model 1 (G1) --------
X1 = df[['G1']]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2)

model1 = LinearRegression()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)
score1 = r2_score(y_test, pred1)
print(score1)


# -------- Model 2 (G2) --------
X2 = df[['G2']]

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)

model2 = LinearRegression()
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)
score2 = r2_score(y_test, pred2)
print(score2)


#------model3-----

X3=df[['G1','G2','studytime']]

X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.2)

model3=LinearRegression()
model3.fit(X_train, y_train)

pred3= model3.predict(X_test)
score3 = r2_score(y_test, pred3)
print(pred3[:5])   # predicted
print(y_test[:5])  # actual
print(score3)




# -------- Compare --------
print("G1 Score:", score1)
print("G2 Score:", score2)

if score1 > score2:
    if score3 >score1:
        print("new r2 is better")
    else:
       print("G1 is better")
else:
    if score3 > score2:
        print("new r2 is better")
    else:
        print("G2 is better")

#------ Q5-----
df['pass']=0
df.loc[df['G3']>=10,'pass']=1
print(df.head(5))

print(len(df.loc[df['pass']==1]))

print(len(df.loc[df['pass']==0]))