import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('dataset/bitext-hospitality-llm-chatbot-training-dataset.csv')

# convert to lowercase
df['instruction'] = df['instruction'].str.lower()

# shuffle data
df = df.sample(frac=1).reset_index(drop=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['instruction'])
y = df['intent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

responses = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free WiFi, breakfast, pool, gym, and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre.",
    "ask_checkin_time": "Check-in time is from 2:00 PM.",
    "ask_checkout_time": "Check-out time is at 12:00 PM.",
    "ask_booking": "You can book directly through our website or at the front desk.",
    "ask_cancellation": "Cancellations are free up to 24 hours before arrival.",
    "greeting": "Hello! How may I assist you today?",
    "goodbye": "Goodbye! Have a great day!",
}

def chatbot_reply(user_input):
    user_input = user_input.lower()
    vector = vectorizer.transform([user_input])
    intent = model.predict(vector)[0]
    return intent

# Test
prompt = ""
while prompt.lower() != "exit":
    prompt = input("You: ")
    if prompt.lower() == "exit":
        print("Chatbot: Goodbye! Have a great day!")
        break
    intent = chatbot_reply(prompt)
    response = responses.get(intent, "I'm sorry, I don't understand your question.")
    print("Chatbot:", response)
