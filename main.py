import openai
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/fit_score', methods=['POST'])
def fit_score():
    data = request.get_json()
    resume = data.get("resume_text")
    job_desc = data.get("job_description_text")

    if not resume or not job_desc:
        return jsonify({"error": "Missing resume or job description"}), 400

    try:
        res_embed = get_embedding(resume)
        jd_embed = get_embedding(job_desc)

        similarity = cosine_similarity(res_embed, jd_embed)
        score = round(similarity * 100)

        return jsonify({
            "fit_score": score,
            "match_quality": (
                "Excellent" if score > 85 else
                "Strong" if score > 70 else
                "Moderate" if score > 50 else
                "Weak"
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Required for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

