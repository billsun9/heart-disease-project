from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET", "POST"])
def result():
    form = request.form
    if request.method == 'POST':
        print(request)
        return render_template("results.html", pred=form['year'])
if __name__ == "__main__":
    app.run(debug=False)