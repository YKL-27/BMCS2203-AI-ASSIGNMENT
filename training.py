print("Loading datasets...")
import pandas as pd
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.data import find
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#! Load both datasets and concatenate them into a single dataframe
df1 = pd.read_csv('data/bitext-hospitality-llm-chatbot-training-dataset.csv')
df2 = pd.read_csv('data/custom-dataset.csv')
df = pd.concat([df1, df2], ignore_index=True)
print(f"Total records: {len(df)}")

# Download NLTK resources
def safe_nltk_download(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

safe_nltk_download('corpora/wordnet')
safe_nltk_download('corpora/omw-1.4')

#! Data preparation
print("Preparing data...")
# convert to lowercase
df['instruction'] = df['instruction'].str.lower()
# shuffle data
df = df.sample(frac=1).reset_index(drop=True)
# vectorise: text -> numerical vector
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.9
)
X = vectorizer.fit_transform(df['instruction'])
y = df['intent']
# Lemmatization
lemmatizer = WordNetLemmatizer()
df['instruction'] = df['instruction'].apply(
    lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split())
)


#! Data training with logistic regression algorithm
print("Training model...")
# train-test split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000,
                            solver='lbfgs',
                            C=3.0,
                            penalty='l2')
model.fit(X_train, y_train)
print("Training completed.")

#! Model evaluation
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

#! Save model and vectorizer
print("Saving model...")
joblib.dump(model, "joblib/intent_model.joblib")
joblib.dump(vectorizer, "joblib/intent_vectorizer.joblib")
print("Saved.")
