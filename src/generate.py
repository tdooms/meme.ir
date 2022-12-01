import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling

train = pd.read_feather("data/train.feather")
test = pd.read_feather("data/test.feather")

train = train.sample(100)
test = test.sample(10)

train["text"] = train["boxes"].apply(lambda x: ". ".join(x))
test["text"] = test["boxes"].apply(lambda x: ". ".join(x))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained('gpt2')

train = Dataset.from_list([{"text": x} for x in train["text"]])
test = Dataset.from_list([{"text": x} for x in test["text"]])

block_size = 128


def preprocess(examples): return tokenizer(examples["text"], truncation=True)


def group_texts(examples):
    print(examples)
    return None


train_tokenized = train.map(preprocess, batched=True, remove_columns=["text"])
test_tokenized = test.map(preprocess, batched=True, remove_columns=["text"])

train_lm = train_tokenized.map(group_texts, batched=True)
# test_lm = test_tokenized.map(group_texts, batched=True)

# print(train_lm)
# print(test_lm)

# training_args = TrainingArguments(
#     output_dir="results",
#     learning_rate=2e-5,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )
#
# collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_lm,
#     eval_dataset=test_lm,
#     tokenizer=tokenizer,
# )
#
# trainer.train()
# trainer.evaluate()
