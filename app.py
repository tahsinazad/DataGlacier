from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

pack = joblib.load("model.pkl")
model = pack["model"]
feature_names = pack["feature_names"]
target_names = pack["target_names"]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Read from form
        sl = float(request.form.get("sepal_length", 0))
        sw = float(request.form.get("sepal_width", 0))
        pl = float(request.form.get("petal_length", 0))
        pw = float(request.form.get("petal_width", 0))
        X = np.array([[sl, sw, pl, pw]])
        pred_idx = model.predict(X)[0]
        prediction = str(target_names[pred_idx])
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X = np.array([[
        float(data.get("sepal_length", 0)),
        float(data.get("sepal_width", 0)),
        float(data.get("petal_length", 0)),
        float(data.get("petal_width", 0)),
    ]])
    pred_idx = model.predict(X)[0]
    return jsonify({"prediction": str(target_names[pred_idx])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
