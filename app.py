import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
app = Flask(__name__)
MODEL_PATH = "model/titanic_model.joblib"
model = joblib.load(MODEL_PATH)
feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            input_dict = {}
            for f in feature_names:
                val = request.form[f]
                if f in ['Pclass', 'Age', 'Fare']:  
                    val = float(val)
                input_dict[f] = [val] 
            input_df = pd.DataFrame(input_dict)
            pred = model.predict(input_df)[0]
            result = "Survived" if pred == 1 else "Not Survived"
        except Exception as e:
            result = f"Error: {e}"
    return render_template("index.html", result=result)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
