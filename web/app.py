import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
import pandas as pd
from src.data_loader import load_movielens_data
from src.preprocessing import filter_data
from src.collaborative_filtering import train_svd_model
from src.topn_recommendation import get_top_n

app = Flask(__name__)

# Load data
ratings = pd.read_csv("data/ratings.csv")
model, predictions = train_svd_model(ratings)
top_n = get_top_n(predictions, n=5)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form["userId"]
    try:
        user_id = int(user_id)
        recommendations = top_n.get(user_id, [])
        return render_template("results.html", recommendations=recommendations)
    except Exception as e:
        return render_template("results.html", recommendations=[], error=str(e))

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    user_id = int(request.args.get("userId"))
    recommendations = top_n.get(user_id, [])
    return jsonify(recommendations=[
        {"movieId": mid, "score": round(score, 2)} for mid, score in recommendations
    ])

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
