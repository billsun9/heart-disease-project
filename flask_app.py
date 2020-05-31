from flask import render_template, Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/results", methods=["POST"])
def results():
    form = request.form
    if request.method == "POST":
        model = joblib.load('models/gaussianModel.joblib')
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
        input = np.array((float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)))
        input = np.expand_dims(input, axis=0)
        prediction = model.predict(input)
        return render_template("results.html", pred = prediction)
if __name__ == '__main__':
    app.run(debug=False)