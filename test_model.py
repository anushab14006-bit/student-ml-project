import joblib

model = joblib.load('model.pkl')

sample = [[10, 12, 2]]  # G1, G2, studytime
result = model.predict(sample)

print(result)