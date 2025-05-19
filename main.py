from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("model.pkl")  # Trained genre classifier
tfidf = joblib.load("tfidf.pkl")  # Trained TF-IDF vectorizer
mlb = joblib.load("mlb.pkl")      # MultiLabelBinarizer for genres

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
def predict(request: Request, overview: str = Form(...)):
    vectorized = tfidf.transform([overview])
    prediction = model.predict(vectorized)
    labels = mlb.inverse_transform(prediction)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": ", ".join(labels[0]) if labels[0] else "None"
    })
