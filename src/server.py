import glob

from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer

app = Flask(__name__)

path = "models/model500k"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(path)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@app.route("/generate", methods=["POST"])
def generate():
    data = str(request.data)
    results = pipeline(data, truncation=True, top_k=5)

    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/templates", methods=["GET"])
def templates():
    results = glob.glob("frontend/templates/*")
    results = [template[19:] for template in results]  # frontend/templates\\

    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
