import joblib

import numpy as np
import pandas as pd

model = joblib.load('models/gaussianModel.joblib')
# %%
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pd.read_csv('data.csv', header=None, names=names)
data['num'] = data['num'].map({0:0, 1:1, 2:1, 3:1, 4:1})
data = data.replace(to_replace="?", value=np.NaN)
#
data['ca'] = data['ca'].astype(float)
ca_mean = data.loc[~(data['ca'].isnull())]['ca'].mean()
data['ca']=data['ca'].fillna(round(ca_mean))
#
data['thal'] = data['thal'].astype(float)
thal_mean = data.loc[~(data['thal'].isnull())]['thal'].mean()
data['thal']=data['thal'].fillna(round(thal_mean))

dataset = data.values
X = dataset[:,:13]
# %%
l = 1
b = 2
j=20
c = np.array((l,b,j))
print(c)
# %%
test = X[4]
test = np.expand_dims(test, axis=0)
print(model.predict(test))