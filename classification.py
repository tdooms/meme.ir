import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


name = "distilbert-base-uncased"
path = "model1"
tokenizer = AutoTokenizer.from_pretrained(name)


def train_model():
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    memes = pd.read_feather("data/memes.feather")
    statistics = pd.read_csv("data/statistics.csv")

    memes["text"] = memes["boxes"].apply(lambda x: ". ".join(x))
    memes = memes[["text", "label"]]
    memes = memes.sample(1_000).reset_index(drop=True)

    mask = np.random.rand(len(memes)) < 0.8
    train = memes[mask]
    test = memes[~mask]

    train = Dataset.from_pandas(train, split="train")
    test = Dataset.from_pandas(test, split="test")

    def preprocess(examples): return tokenizer(examples["text"], truncation=True)

    train_tokenized = train.map(preprocess, batched=True)
    test_tokenized = test.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=100)

    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        label_names=statistics["path"].str.strip(".json").replace('-', ' ').tolist(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(path)


def test_model():
    model = AutoModelForSequenceClassification.from_pretrained(path)

    text = [
        "One does not simply generate memes with AI",  # label 1
        "Y U NO USE RUST?"  # label 17
    ]

    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k, v in encoding.items()}

    outputs = model(**encoding)
    print(outputs)


if __name__ == '__main__':
    train_model()
    test_model()
