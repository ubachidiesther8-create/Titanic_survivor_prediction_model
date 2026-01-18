from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        # Collect form data
        pclass = int(request.form["pclass"])
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        sex = request.form["sex"]
        embarked = request.form["embarked"]

        # Create DataFrame (IMPORTANT: column names must match training data)
        input_data = pd.DataFrame([{
            "Pclass": pclass,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Sex": sex,
            "Embarked": embarked
        }])

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        prediction = "Survived" if pred == 1 else "Did Not Survive"
        probability = round(prob * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
