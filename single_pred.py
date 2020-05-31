import joblib
model = joblib.load("models/gaussianModel.joblib")
# %%
import numpy as np
x1 = X[0]
x1 = np.expand_dims(x1, axis=0)
# 
y1 = model.predict(x1)
# 
# print(np.array_equal(np.array([1], dtype='int32'), y1.astype('int32')))