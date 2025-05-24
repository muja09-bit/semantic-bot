from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)

# Load model ringan
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

# Precompute embeddings untuk semua soalan
questions = [item["question"] for item in knowledge]
question_embeddings = model.encode(questions, convert_to_tensor=True)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"reply": "Sorry, I didn’t receive any message."}), 400

        # Cari soalan paling hampir
        user_embedding = model.encode(user_message, convert_to_tensor=True)
        scores = util.cos_sim(user_embedding, question_embeddings)[0]
        best_idx = int(scores.argmax())

        confidence = float(scores[best_idx])
        if confidence >= 0.6:
            reply = knowledge[best_idx]["answer"]
        else:
            reply = "I'm sorry, I don't know about that. Can you teach me?"

        return jsonify({"reply": reply})
    
    except Exception as e:
        return jsonify({"reply": f"(Server error) {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
