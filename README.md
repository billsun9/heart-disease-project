# heart-disease-project

Heart disease is the leading cause of death in first world countries like the United States, the United Kingdom, and Canada. According the the CDC, nearly one in every four deaths in the US can be attributed to heart disease.

In this project, I trained multiple machine learning models to diagnose a patient's for risk of heart disease. I used the scikit-learn library in Python to develop 6 different models(Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors Classifier, Decision Tree Classifier, Support Vector Machine (SVM), and Gaussian Naive Bayes Classifier).

The heart disease dataset (data.csv) used for training was collected by the Long Beach and Cleveland Clinic Foundation. The associated code for training and testing the ML models is located in train_1.py and train_2.py

To allow for greater user accessibility, I hosted these models on a web application using the Flask web-framework library in Python. The HTML for the website is in the templates directory, and the HTML and CSS is in the static directory. 

The web application is located at https://hearthelper.pythonanywhere.com/

