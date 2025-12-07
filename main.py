import json
import joblib
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "main", "intent_model.joblib")
INTENT_VECTORIZER_PATH = os.path.join(BASE_DIR, "main", "intent_vectorizer.joblib")
RESPONSES_PATH = os.path.join(BASE_DIR, "data", "responses.json")

# Load model + vectorizer
model = joblib.load(INTENT_MODEL_PATH)
vectorizer = joblib.load(INTENT_VECTORIZER_PATH)

# Load responses
with open(RESPONSES_PATH, "r") as f:
    responses = json.load(f)

def chatbot_reply(user_input):
    user_input = user_input.lower()
    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]
    response = responses.get(intent, responses.get("unknown_intent"))
    if isinstance(response, list):
        response = response[0]
    return response

# Streamlit app
def main():
    st.set_page_config(page_title="Intent Chatbot", page_icon="🤖")

    st.title("🤖 Intent Recognition Chatbot")

    # Keep conversation state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Chatbot:** {msg['text']}")

    # User input
    user_input = st.text_input("Type your message:")

    if st.button("Send") and user_input.strip() != "":
        # Add user message
        st.session_state.messages.append({"role": "user", "text": user_input})

        # Generate reply
        reply = chatbot_reply(user_input)
        st.session_state.messages.append({"role": "bot", "text": reply})

        # Force app to rerun
        st.experimental_rerun()

    # Reset chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == '__main__':
    main()
