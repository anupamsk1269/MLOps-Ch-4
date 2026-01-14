#import os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from transformers import AutoTokenizer

app = Flask(__name__)

# Load Tokenizer and ONNX Session once at startup
MODEL_PATH = "models/roberta.onnx"
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
session = ort.InferenceSession(MODEL_PATH)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.route('/')
def health():
    return jsonify({"status": "ready", "model": "roberta-onnx"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # 1. Preprocess
    inputs = tokenizer(data['text'], return_tensors="np")
    
    # 2. Run Inference
    # Map 'input_ids' and 'attention_mask' to the ONNX names
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    outputs = session.run(None, onnx_inputs)
    
    # 3. Postprocess (Softmax to get probabilities)
    scores = softmax(outputs[0])
    labels = ["Negative", "Neutral", "Positive"]
    ranking = np.argsort(scores[0])[::-1]
    
    results = [
        {"label": labels[i], "score": float(scores[0][i])} 
        for i in ranking
    ]

    return jsonify({"text": data['text'], "predictions": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)