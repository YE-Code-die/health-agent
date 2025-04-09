from flask import Flask, render_template, request, jsonify
import torch 
from query_agent import ask_question
import os

app = Flask(__name__)

# Set model configuration
MODEL_TYPE = "huggingface"
MODEL_PATH = "facebook/opt-1.3b"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        answer = ask_question(question, model_type=MODEL_TYPE, model_path=MODEL_PATH)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # print(torch.__version__)
    # print(torch.rand(2, 3))
    app.run(debug=True) 
