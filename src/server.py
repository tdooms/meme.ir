from flask import Flask, request
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer

app = Flask(__name__)

path = "models/model500k"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(path)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@app.route("/generate", methods=["POST"])
def generate():
    results = pipeline(request.json["text"], truncation=True, top_k=5)
    print(results)
    return "<p>Hello, World!</p>"
