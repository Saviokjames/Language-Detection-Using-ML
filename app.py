import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your trained model and CountVectorizer
model = joblib.load("trained_model.pkl")
cv = joblib.load("count_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_language = None

    if request.method == "POST":
        user_input = request.form.get("text")

        if user_input:
            # Transform user input using the CountVectorizer
            input_data = cv.transform([user_input]).toarray()

            # Predict the language
            predicted_language = model.predict(input_data)[0]

    return render_template("index.html", prediction=predicted_language)

if __name__ == "__main__":
    app.run(debug=True)
