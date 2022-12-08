import glob
import time

import pandas as pd
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

    results = [{**result, "path": f"templates/{result['label'].replace(' ', '-')}.jpg"} for result in results]
    print(results)

    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/templates", methods=["GET"])
def templates():
    results = pd.read_csv("data/statistics.csv")["path"].values.tolist()
    results = [{"label": x[:-5].replace('-', ' '), "path": f'templates/{x.replace(".json", ".jpg")}'} for x in results]
    print(results)

    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
