import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
from preprocess import clean_text

# Load your local CSV
df = pd.read_csv("dataset/train.csv")

# Combine Title + Description
df["text"] = df["Title"] + " " + df["Description"]

# Convert 1–4 → 0–3
df["label"] = df["Class Index"] - 1

# Clean the text column
df["clean_text"] = df["text"].apply(clean_text)

# Define X and y
X = df["clean_text"]
y = df["label"]

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
'''
What is max_features in TfidfVectorizer?
✔ It limits the number of words (features) used by the TF-IDF vectorizer.

TF-IDF normally converts text into a matrix like:

sentence → [word1_count, word2_count, word3_count, ...]


If your dataset has 80,000 unique words, TF-IDF will create 80,000 columns → huge, slow, heavy model.

To avoid this, we use:

tfidf = TfidfVectorizer(max_features=5000)


Meaning:

👉 Keep only the top 5000 most important words
👉 Ignore rare, uncommon, unnecessary words
👉 Reduce training time
👉 Reduce dimensionality
👉 Prevent overfitting

🔍 Example:

Imagine your dataset has these words:

the, is, cricket, economy, ai, politics, stock, bitcoin, ...


TF-IDF may find that 30,000 unique words exist.

With max_features=5000:

✔ The top 5000 words (most frequent across dataset) are used
✔ Rare words like "xyzabc", "lolll", "####", "wirwui" are removed

This improves:

speed

memory

model accuracy
'''


# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
