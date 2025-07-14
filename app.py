import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form['text']
        # Predict using your model
        vector = vectorizer.transform([input_text])
        prediction = model.predict(vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

    return render_template('index.html', sentiment=sentiment, input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)
