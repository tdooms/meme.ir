from flask import Flask, request
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline, AutoTokenizer

app = Flask(__name__)

path = "models/model500k"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(path, output_hidden_states=True)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@app.route("/generate", methods=["POST"])
def hello_world():
    results = pipeline(request.json["text"])
    print(results)
    return "<p>Hello, World!</p>"
