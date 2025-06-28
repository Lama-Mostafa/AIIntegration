from flask import Flask,jsonify,request
from flask_cors import CORS
#from chatbot import rag, collection, llm_model
from chatbot import generate_response

app = Flask(__name__)
CORS(app)
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        print("hjasgjfdsalkgjosf")
        answer, context, score = generate_response(question)

        return jsonify({
            "answer": answer,
            "context": context,
            "similarity_score": score
        })
    except Exception as e:
        return jsonify({
            "message":repr(e)
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
