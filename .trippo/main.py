# from transformers import pipeline
# from flask import Flask, request, jsonify

# # Load NLP chatbot model
# chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

# # Initialize Flask app
# app = Flask(__name__)

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.json.get("message", "")
#     response = chatbot(user_input, max_length=100)[0]["generated_text"]
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)

