from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
vect = pickle.load(open("vect.pkl", "rb"))

@app.route("/")
def index():
    return render_template("abc.html")

@app.route('/page1')
def page1():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        text = request.form["text"]
        text_vectorized = vect.transform([text])
        prediction = model.predict(text_vectorized)
        return render_template("index.html", prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)