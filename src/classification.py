import os.path
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# I've moved the imports into the functions so that they don't get imported when I'm just running the script
# For some reason some imports take 3+ seconds
name = "distilbert-base-uncased"
path = "models/model500k"


def train_model(tokenizer):
    from datasets import Dataset
    from transformers import DataCollatorWithPadding
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

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


def visualise_model(tokenizer):
    from transformers import AutoModelForSequenceClassification

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
        memes = pd.read_feather("data/subset.feather")

    features = embeddings.sum(axis=1)
    features = TSNE(n_components=2).fit_transform(features)

    print(features)
    print(memes['label'].values)

    # matrix = cosine_similarity(features)
    # sns.heatmap(matrix)
    # plt.show()

    memes = pd.merge(memes, pd.read_csv("data/statistics.csv"), on="label")

    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=memes['category'].values)
    plt.show()


def infer_model(tokenizer, first=False):
    from transformers import TextClassificationPipeline
    from transformers import AutoModelForSequenceClassification

    infer_path = "first" if first else "inferred"

    if not os.path.exists(f"data/{infer_path}.feather"):
        model = AutoModelForSequenceClassification.from_pretrained(path)
        memes = pd.read_feather("data/test.feather")
        memes = memes.sample(10_000).reset_index(drop=True)

        func = lambda l: l[0] if first else lambda l: ". ".join(l)
        text = memes["boxes"].apply(func).tolist()

        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        # results = pipeline(text, truncation=True, top_k=None) # This sorts the logits which we don't really want
        results = pipeline(text, truncation=True, return_all_scores=True)

        logits = [[x['score'] for x in row] for row in results]
        frame = pd.DataFrame(logits, columns=[f"logit_{i}" for i in range(99)])
        memes = pd.concat([memes, frame], axis=1)

        memes.to_feather(f"data/{infer_path}.feather")
    else:
        print("Already inferred, please remove the file explicitly if you want to re-infer")


def explain_model(first=False):
    infer_path = "first" if first else "inferred"
    memes = pd.read_feather(f"data/{infer_path}.feather")
    statistics = pd.read_csv("data/statistics.csv")

    matrix = memes[[f"logit_{i}" for i in range(99)]].to_numpy()
    memes["guessed"] = matrix.argmax(axis=1)

    memes["correct"] = memes["label"] == memes["guessed"]
    memes = pd.merge(memes, statistics, on='label', how="left")

    # print(memes[memes["label"] == 98])

    # Accuracy on average and per class
    print(f"average accuracy:", len(memes[memes['correct']]) / len(memes))
    accuracy = memes.groupby("label")["correct"].apply(lambda x: sum(x) / len(x)).reset_index(name="accuracy")
    statistics = pd.merge(statistics, accuracy, on='label', how="left")
    print(accuracy)

    # Scatterplot of accuracy and sample count
    sns.scatterplot(x=statistics["count"], y=statistics["accuracy"], hue=statistics["category"])
    plt.show()

    # Barplot of accuracy per category
    sns.barplot(x="category", y="correct", data=memes)
    plt.show()

    # Confusion matrix of labels
    # confusion = memes.groupby(["guessed", "label"]).size().unstack(fill_value=0)
    # normalized = confusion / confusion.sum()
    # sns.heatmap(normalized)
    # plt.show()


def wrong_samples():
    memes = pd.read_feather("data/inferred.feather")

    matrix = memes[[f"logit_{i}" for i in range(99)]].to_numpy()
    memes["guessed"] = matrix.argmax(axis=1)

    memes["correct"] = memes["label"] == memes["guessed"]
    rows = memes[(memes["correct"] == 0) & (memes["label"] == 9)]
    rows = rows.sort_values("votes", ascending=False)
    print(rows.head(10))


def test_model(tokenizer):
    from transformers import TextClassificationPipeline
    from transformers import AutoModelForSequenceClassification

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


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)

    pd.options.display.max_rows = 100
    pd.options.display.max_columns = None
    pd.options.display.width = None

    # train_model(tokenizer)
    # infer_model(tokenizer, True)
    # explain_model(True)
    # wrong_samples()
    visualise_model(tokenizer)
    # test_model(tokenizer)


if __name__ == '__main__':
    main()
