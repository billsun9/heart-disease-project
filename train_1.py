import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
Only 14 attributes used:
1. #3 (age)
2. #4 (sex)
3. #9 (cp)
4. #10 (trestbps)
5. #12 (chol)
6. #16 (fbs)
7. #19 (restecg)
8. #32 (thalach)
9. #38 (exang)
10. #40 (oldpeak)
11. #41 (slope)
12. #44 (ca)
13. #51 (thal)
14. #58 (num) (the predicted attribute)
'''
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pd.read_csv('data.csv', header=None, names=names)
#%% data preprocessing
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

# %% train/val split
from sklearn.model_selection import train_test_split
dataset = data.values
X = dataset[:,:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=1)
# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# %%
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# %%
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
# %%
# Evaluate predictions
print(accuracy_score(Y_test, predictions))
# %%
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
# %%
# save model
import joblib
joblib.dump(model, 'models/gaussianModel.joblib')