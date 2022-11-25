import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, TextClassificationPipeline, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


name = "distilbert-base-uncased"
path = "models/model500k"
tokenizer = AutoTokenizer.from_pretrained(name)


def train_model():
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train = pd.read_feather("data/train.feather")
    test = pd.read_feather("data/test.feather")

    statistics = pd.read_csv("data/statistics.csv")

    # memes = memes.sample(10_000).reset_index(drop=True)

    train["text"] = train["boxes"].apply(lambda x: ". ".join(x))
    test["text"] = test["boxes"].apply(lambda x: ". ".join(x))

    train = train[["text", "label"]]
    test = test[["text", "label"]]

    train = Dataset.from_pandas(train, split="train")
    test = Dataset.from_pandas(test, split="test")

    def preprocess(examples): return tokenizer(examples["text"], truncation=True)

    train_tokenized = train.map(preprocess, batched=True)
    test_tokenized = test.map(preprocess, batched=True)

    statistics["name"] = statistics["path"].str[:-5].replace('-', ' ', regex=True)
    id2label = {k: v for k, v in statistics[["label", "name"]].values}

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=99, id2label=id2label)

    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
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


def visualise_model():
    if not os.path.exists("data/embeddings.npy"):
        memes = pd.read_feather("data/test.feather")
        memes = memes.sample(1_000).reset_index(drop=True)
        memes.to_feather("data/subset.feather")

        text = memes["boxes"].apply(lambda x: ". ".join(x)).tolist()

        model = AutoModelForSequenceClassification.from_pretrained(path)
        encoding = tokenizer(text, return_tensors='pt', padding=True)["input_ids"]
        outputs = model(encoding, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].detach().cpu().numpy()
        np.save('data/embeddings.npy', embeddings)
    else:
        embeddings = np.load('data/embeddings.npy')
        memes = pd.read_feather("subset.feather")

    features = embeddings.sum(axis=1)
    features = PCA(n_components=2).fit_transform(features)

    print(features)
    print(memes['label'].values)

    # matrix = cosine_similarity(features)
    # sns.heatmap(matrix)
    # plt.show()

    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=memes['label'].values)
    plt.show()


def explain_model():
    if not os.path.exists("data/inferred.feather"):
        model = AutoModelForSequenceClassification.from_pretrained(path)
        memes = pd.read_feather("data/test.feather")
        # memes = memes.sample(100).reset_index(drop=True)

        text = memes["boxes"].apply(lambda x: ". ".join(x)).tolist()

        statistics = pd.read_csv("data/statistics.csv")
        statistics["name"] = statistics["path"].str[:-5].replace('-', ' ', regex=True)
        label2id = {v: k for k, v in statistics[["label", "name"]].values}

        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        results = pipeline(text, truncation=True)

        memes["predicted"] = [label2id[result["label"]] for result in results]
        memes["confidence"] = [result["score"] for result in results]
        memes["correct"] = memes["label"] == memes["predicted"]

        memes.to_feather("data/inferred.feather")
    else:
        memes = pd.read_feather("data/inferred.feather")

    print(len(memes))
    print(memes["correct"].value_counts())
    print(memes.groupby("label")["correct"].apply(lambda x: sum(x) / len(x)))


def test_model():
    model = AutoModelForSequenceClassification.from_pretrained(path, output_hidden_states=True)

    text = [
        "One does not simply generate memes with AI",
        "Y U NO USE RUST?",
        "This is where I'd put my AI image. If I had one!",
        "This is where I'd put my anal beads. If I had one!",
        "One does not simply trust the labels given by the AI",
        "Do you want to get killed? Cause that how you get killed",
        "Programming in Java. Programming in C, Programming in Assembly",
    ]

    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    results = pipeline(text)
    print(results)


if __name__ == '__main__':
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = None
    pd.options.display.width = None

    # train_model()
    explain_model()
    # visualise_model()
    # test_model()
