import gradio as gr
import pickle

# Load model and vectorizer
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_sentiment(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Positive " if prediction == 1 else "Negative "

# Gradio interface
iface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text", title="Twitter Sentiment Analyzer")

if __name__ == "__main__":
    iface.launch()
