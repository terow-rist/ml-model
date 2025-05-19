import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import joblib
import ast

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/movies_metadata.csv", low_memory=False)
keywords_df = pd.read_csv("data/keywords.csv")

df['id'] = df['id'].astype(str)
keywords_df['id'] = keywords_df['id'].astype(str)

df = df[['id', 'overview', 'genres']].dropna(subset=['overview', 'genres'])
df = df[df['genres'].str.len() > 2]
df = df.merge(keywords_df[['id', 'keywords']], on='id', how='left')

def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

# Extract keyword names
def extract_keywords(keyword_str):
    try:
        keywords = ast.literal_eval(keyword_str)
        return " ".join([kw['name'].replace(" ", "_") for kw in keywords])
    except:
        return ""

df['genres'] = df['genres'].apply(extract_genres)
df['keywords'] = df['keywords'].fillna("").apply(extract_keywords)
df['overview'] = df['overview'] + " " + df['keywords']
df = df[df['overview'].str.len() > 20]

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['overview'])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(model, "models/model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(mlb, "models/mlb.pkl")
