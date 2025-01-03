# file: webapp/app.py

"""
Flask app with a hybrid approach:
1) Use the loaded classification pipeline to predict if each candidate is Disqualified=1 or 0.
2) Filter out disqualified (1).
3) On the remaining candidates, compute TF-IDF similarity with the user-provided job description.
4) Sort top 30 by similarity (0-100).
"""


from typing import Any

from flask import Flask, request, render_template, jsonify


from flask_cors import CORS

# Import the classification pipeline loader
from model.job_descriptions import (
    compute_similarity,
    filter_candidates,
    init_candidate_data,
    validate_job_description,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    """
    Renders an index page with a form for the job description.
    """
    return render_template("index.html")


@app.route("/", methods=["POST"])
def process_candidates() -> Any:
    """
    Hybrid approach to filter and rank candidates based on job description similarity.
    1. Predict disqualification using the classification pipeline.
    2. Compute TF-IDF similarity for non-disqualified candidates.
    3. Return top 30 candidates sorted by similarity.
    """
    data = request.get_json()
    job_desc = data.get("job_description", "").strip()

    if not validate_job_description(job_desc):
        return (
            jsonify(
                {"message": "Please provide a valid Job Description (1-3500 chars)."}
            ),
            400,
        )

    df_filtered = filter_candidates()
    if df_filtered.empty:
        return jsonify({"job_description": job_desc, "candidates": []})

    top_candidates = compute_similarity(df_filtered, job_desc)
    return jsonify({"job_description": job_desc, "candidates": top_candidates})


if __name__ == "__main__":
    init_candidate_data()
    app.run(debug=True)
