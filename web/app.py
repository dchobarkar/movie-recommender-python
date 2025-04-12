from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form["userId"]
    recommendations = model.get(user_id, [])  # Simulated logic
    return render_template("results.html", recommendations=recommendations)

@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    user_id = request.args.get("userId")
    recommendations = model.get(user_id, [])  # Simulated logic
    return jsonify(recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
