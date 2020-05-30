from flask import render_template, Flask, request

app = Flask(__name__)
@app.route("/")
def index():
    return "<h1>Hello world!</h1>"

if __name__ == '__main__':
    app.run(debug=False)