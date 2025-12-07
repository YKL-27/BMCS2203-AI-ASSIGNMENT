import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#! load both datasets and combine them
print("Loading datasets...")
df1 = pd.read_csv('bitext-hospitality-llm-chatbot-training-dataset.csv')
df2 = pd.read_csv('newdataset.csv')

df = pd.concat([df1, df2], ignore_index=True)
print(f"Total records: {len(df)}")

print("Preparing data...")
#! data preparation
# convert to lowercase
df['instruction'] = df['instruction'].str.lower()
# shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# vectorise: text -> numerical vector
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['instruction'])
y = df['intent']

#! data training
print("Training model...")
# train-test split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#! model evaluation
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

#! load responses
with open('dataset/responses.json', 'r') as f:
    responses = json.load(f)

def chatbot_reply(user_input):
    user_input = user_input.lower()
    vector = vectorizer.transform([user_input])
    intent = model.predict(vector)[0]
    return intent

#! save model and vectorizer
print("Saving model...")
joblib.dump(model, "intent_model.joblib")
joblib.dump(vectorizer, "intent_vectorizer.joblib")
print("Saved.")
