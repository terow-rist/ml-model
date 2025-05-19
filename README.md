# 🎬 Movie Genre Prediction

A simple web application that predicts movie genres based on an input **overview** using **TF-IDF vectorization** and a **multi-label Logistic Regression model**. Built with **FastAPI** and rendered using **Jinja2 templates**.

---

## 🧠 Overview

This project demonstrates a **multi-label text classification** pipeline:

* A dataset of movie overviews and genres is preprocessed.
* Text features are extracted using `TfidfVectorizer`.
* Genres are binarized using `MultiLabelBinarizer`.
* A `LogisticRegression` model is trained with `OneVsRestClassifier`.

The trained model is served via a FastAPI backend where users can input a custom movie overview to get predicted genres.

---

## 🚀 Features

* Multi-label classification using Scikit-learn
* Trained on TMDB’s `movies_metadata.csv`
* FastAPI backend with Jinja2 HTML templates
* Styled HTML form for overview input and genre output

---

## 📦 Requirements

Install Python packages:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```txt
fastapi
uvicorn
jinja2
scikit-learn
pandas
joblib
```

---

## 🗃️ Dataset

Uses the [movies\_metadata.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) file from TMDB. Only `overview` and `genres` columns are used.

---

## 🛠️ Usage

### 1. Train the Model

```bash
python train.py
```

This will:

* Clean and vectorize the dataset
* Train a Logistic Regression model for genre prediction
* Save the model, TF-IDF vectorizer, and label binarizer as `.pkl` files

### 2. Run the Web App

```bash
uvicorn main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🌐 Web UI

* Enter a movie overview (e.g. "A team of explorers travel through a wormhole in space.")
* Click **Predict**
* Get predicted genres like: `Science Fiction, Adventure`

---

## 📁 Project Structure

```
.
├── main.py              # FastAPI app
├── train.py             # Model training script
├── model.pkl            # Trained model
├── tfidf.pkl            # TF-IDF vectorizer
├── mlb.pkl              # MultiLabelBinarizer
├── templates/
│   └── index.html       # Jinja2 template
├── movies_metadata.csv  # Dataset
└── README.md
```

---

## 📌 Example Predictions

| Overview                                      | Predicted Genres           |
| --------------------------------------------- | -------------------------- |
| "A spaceship lands on a mysterious planet..." | Science Fiction, Adventure |
| "Two lovers separated by war..."              | Drama, Romance             |
| "A hilarious journey of two friends..."       | Comedy                     |

---

## 📄 License

This project is for educational/demo purposes. The dataset is from TMDB via Kaggle.

---

Let me know if you want to include a GIF or screenshot of the UI in the README.
