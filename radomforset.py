import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

# load data
df = pd.read_csv('student_data.csv')

# create pass column
df['pass'] = 0
df.loc[df['G3'] >= 10, 'pass'] = 1

# select input and output
X = df[['G1', 'G2','studytime']]
y = df['pass']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=200, max_depth=3)
model.fit(X_train, y_train)


# predictions
pred = model.predict(X_test)

# print predictions
print("Predictions:", pred[:5])



# print actual values
print("Actual:", y_test.values[:5])

# accuracy
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# confusion matrix
con = confusion_matrix(y_test, pred)
print("Confusion Matrix:\n", con)

#saving
import joblib

joblib.dump(model, 'model.pkl')

