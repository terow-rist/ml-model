import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import joblib
import ast

# Load dataset
df = pd.read_csv("data/movies_metadata.csv", low_memory=False)

# Drop missing or invalid values
df = df[['overview', 'genres']].dropna()
df = df[df['genres'].str.len() > 2]

# Extract genre names from JSON-like strings
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

df['genres'] = df['genres'].apply(extract_genres)

# Convert text into TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['overview'])

# Binarize labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save everything
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")  # Instead of scaler
joblib.dump(mlb, "mlb.pkl")
