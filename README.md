# 🎬 Movie Recommender System

This project is a hands-on implementation of a movie recommender system using collaborative filtering (SVD via the `surprise` library) and content-based filtering (TF-IDF + cosine similarity). It includes a lightweight Flask-based web interface for user interaction.

## 🚀 Features

- 📊 Data preprocessing and EDA (MovieLens 100K)
- 🤖 Collaborative Filtering using SVD
- 🧠 Content-Based Filtering via TF-IDF on genres
- 🎯 Top-N Recommendation Generator
- 📈 Evaluation Metrics: RMSE, MAE, Precision@k, Recall@k
- 🌐 Web app with HTML form and API endpoint

## 🗂 Project Structure

```
movie-recommender-python/
├── data/                    # ratings.csv (MovieLens dataset)
├── notebooks/               # EDA and analysis notebooks
├── src/                     # Modular recommender logic
├── web/                     # Flask app and HTML templates
├── requirements.txt         # All dependencies
└── README.md
```

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Flask app

```bash
cd web
python app.py
```

Then open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🧪 Try the API

```bash
curl http://127.0.0.1:5000/api/recommend?userId=1
```

## 📚 Dataset Source

MovieLens 100K  
[https://grouplens.org/datasets/movielens/100k](https://grouplens.org/datasets/movielens/100k)

## 💡 Author

Built by [Darshan Chobarkar](https://github.com/dchobarkar)  
Inspired by [this blog post](https://dchobarkar.github.io/2024/09/21/hands-on-build-a-movie-recommender-in-python.html)
