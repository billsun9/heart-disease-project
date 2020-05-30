import joblib

model = joblib.load('models/gaussianModel.joblib')
# %%
from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
print(accuracy_score(Y_test, predictions))