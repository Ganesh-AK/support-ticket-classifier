import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report
import joblib 
import os


df = pd.read_csv("data/tickets.csv")
x = df["text"]
y = df["category"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
vec = TfidfVectorizer()
x_train_tfidf = vec.fit_transform(x_train)
x_test_tfidf = vec.transform(x_test)

model = LogisticRegression()
model.fit(x_train_tfidf, y_train)
y_pred = model.predict(x_test_tfidf)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

for text, real, guess in zip(x_test, y_test, y_pred):
    if real != guess:
        print(real, "->", guess, ":", text)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(vec, "models/vectorizer.joblib")
