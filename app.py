import streamlit as st
import pickle
import sklearn

# --- APP LAYOUT ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    """Load the vectorizer and sentiment model from disk."""
    try:
        with open('model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        # Return None if files are not found, handle the error outside
        return None, None
    except Exception as e:
        # Also return None for other loading errors
        print(f"An unexpected error occurred loading models: {e}")
        return None, None

vectorizer, model = load_models()

# --- UI ELEMENTS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e0c3fc, #8ec5fc);
    }
    .stTextArea, .stButton {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid #4a90e2;
        background-color: #4a90e2;
        color: white;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        border-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    /* New styles for the prediction box */
    .prediction-box {
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: bold;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }
    .positive {
        background-color: #28a745; /* Darker green */
    }
    .negative {
        background-color: #dc3545; /* Darker red */
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sentiment Analyzer")
st.markdown("<p style='text-align: center;'>Enter some text (like a tweet) to predict its sentiment.</p>", unsafe_allow_html=True)

# Check if models were loaded successfully AFTER setting up the page
if vectorizer is None or model is None:
    st.error("Could not load the model files. Please make sure the 'model' directory with 'vectorizer.pkl' and 'sentiment_model.pkl' exists in the same folder as this script.")
else:
    # --- USER INPUT ---
    with st.form(key='sentiment_form'):
        input_text = st.text_area("Paste the text here:", "", height=150)
        submit_button = st.form_submit_button(label='Analyze Sentiment')

    # --- PREDICTION LOGIC ---
    if submit_button and input_text:
        try:
            # Vectorize the input text
            vector = vectorizer.transform([input_text])

            # Predict the sentiment
            prediction = model.predict(vector)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

            # Display the result using new custom-styled markdown
            st.markdown("---")
            st.write("### Prediction:")
            if sentiment == "Positive":
                st.markdown(f'<div class="prediction-box positive">{sentiment} sentiment detected! </div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box negative">{sentiment} sentiment detected. </div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    elif submit_button and not input_text:
        st.warning("Please enter some text to analyze.")

