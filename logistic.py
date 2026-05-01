import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor

# load data
df = pd.read_csv('student_data.csv')


# create pass column
df['pass'] = 0
df.loc[df['G3'] >= 10, 'pass'] = 1

# select input and output
X = df[['G1', 'G2']]
y = df['pass']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# predictions
pred = model.predict(X_test)

# print predictions
print("Predictions:", pred[:5])

# print actual
print("Actual:", y_test.values[:5])

# accuracy
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

print(df['pass'].value_counts(normalize=True))

con=confusion_matrix(y_test, pred)
print(con)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.Decision_model import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# load data
df = pd.read_csv('student_data.csv')


# create pass column
df['pass'] = 0
df.loc[df['G3'] >= 10, 'pass'] = 1

# select input and output
X = df[['G1', 'G2']]
y = df['pass']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# predictions
pred = model.predict(X_test)

# print predictions
print("Predictions:", pred[:5])

# print actual
print("Actual:", y_test.values[:5])

# accuracy
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

print(df['pass'].value_counts(normalize=True))

con=confusion_matrix(y_test, pred)
print(con)