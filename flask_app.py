from flask import render_template, Flask, request
import joblib
import numpy as np

app = Flask(__name__)
@app.route("/") # home page
def index():
    return render_template('index.html')

@app.route("/results", methods=["POST"]) # shows results of ML prediction and further info
def results():
    form = request.form
    if request.method == "POST":
        # loads the trained gaussian sklearn model
        model = joblib.load('models/gaussianModel.joblib')
        # relevant info from user
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        # reshapes user input
        input = np.array((float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)))
        input = np.expand_dims(input, axis=0) # reshapes from (13,) to (1,13)
        prediction = model.predict(input)
        
        if np.array_equal(np.array([1], dtype='int32'), prediction.astype('int32')):
            pred = "High risk for heart disease"
            message = "The machine learning model predicted that you are at high risk of contracting heart disease. As heart disease is often life-threatening, you should take action to reduce your risk. Although some risk factors, like age, gender, race, and family history cannot be changed, you are able to change your lifestyle."
        else:
            pred = "Low risk for heart disease"
            message = "The machine learning model predicted that you are at low risk of contracting heart disease. Good for you! However, as heart disease is often life-threatening, you should still take action to reduce your risk. Although some risk factors, like age, gender, race, and family history cannot be changed, you are able to change your lifestyle."
        return render_template("results.html", pred = pred, message = message)
if __name__ == '__main__':
    app.run(debug=False)