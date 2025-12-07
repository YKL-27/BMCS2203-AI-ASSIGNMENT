import json
import joblib
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "joblib", "intent_model.joblib")
INTENT_VECTORIZER_PATH = os.path.join(BASE_DIR, "joblib", "intent_vectorizer.joblib")
RESPONSES_PATH = os.path.join(BASE_DIR, "data", "responses.json")

# Load model + vectorizer
try:
    model = joblib.load(INTENT_MODEL_PATH)
    vectorizer = joblib.load(INTENT_VECTORIZER_PATH)
except FileNotFoundError:
    st.error("Model or vectorizer file not found.")
    st.stop()

# Load responses
try:
    with open(RESPONSES_PATH, "r") as f:
        responses = json.load(f)
except FileNotFoundError:
    st.error("Responses file not found.")
    st.stop()

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
    st.title("Chatbot Interface")
    st.caption("Ask me about room prices, check-in times, or facilities!")

    # initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your hotel booking today?"}]

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # handle user input
    if prompt := st.chat_input("Type your message here..."):
        # display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        intent, reply = chatbot_reply(prompt)

        # display chatbot reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    
    
if __name__ == '__main__':
    main()
