import json
import joblib
import time
import numpy as np

#! Load model + vectorizer
try:
    model = joblib.load("joblib/intent_model.joblib")
    vectorizer = joblib.load("joblib/intent_vectorizer.joblib")
except FileNotFoundError:
    print("Model or vectorizer file not found.")

#! Load responses
try:
    with open("data/responses.json", "r") as f:
        responses = json.load(f)
except FileNotFoundError:
    print("Responses file not found.")

#! Chatbot reply function
def chatbot_reply(user_input):
    user_input = user_input.lower()
    start_time = time.time()
    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = float(np.max(proba))
    response = responses.get(intent, "unknown_intent")
    if isinstance(response, list):
        response = response[0]
    elapsed = round((time.time() - start_time) * 1000, 2)
    return {
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "time_ms": elapsed
    }

#! Main chat loop
print("Chatbot Interface")
print("Ask me about room prices, check-in times, or facilities!")
print("Type '>exit' to end the chat.")
prompt = ""
while True:
    prompt = input("\nYou: ")
    if prompt.lower() != ">exit":
        reply = chatbot_reply(prompt)
        print(f"Bot: {reply['response']}\n"
              f"(Intent: {reply['intent']}, Confidence: {reply['confidence']*100:.2f}%, Time: {reply['time_ms']} ms)")
    else:
        break

print("\nChat ended.")
