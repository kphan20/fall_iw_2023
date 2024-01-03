from transformers import BertConfig, BertModel, BertTokenizer, DataCollatorWithPadding, AutoTokenizer
import pickle
import torch
import torch.optim as optim
from model import MultiTaskClassifier

import numpy as np
from datasets import load_metric
import evaluate
from torch import cuda

# https://huggingface.co/blog/sentiment-analysis-python
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess(examples, tokenizer):
    return tokenizer(examples, return_tensors='pt',padding="max_length", truncation=True)

tokenized_train = train_dataset.map(preprocess,batched=True)
tokenized_test = test_dataset.map(preprocess, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
data_collator = DataCollatorWithPadding("""ADD tokenizer=TOKENIZER""")

def compute_metrics(logits, labels):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    metrics = [accuracy, precision, recall, f1]
    results = dict()
    for metric in metrics:
        results.update(metric.compute(predictions=logits, references=labels))
    
    return results

# https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb
device = 'cuda' if cuda.is_available() else 'cpu'

# TODO Find out good train/test split
## TODO figure out which reviews to filter (probably just the shorter ones, maybe try to make ratios of sarcastic/positive combos the same)
## TODO figure out validation split

# Load embedding model
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load steam reviews text and rating
with open("steam_reviews/reviews.pkl", "rb") as f:
    reviews = pickle.load(f)

# Load Vader classification (supposed sarcasm)
with open("steam_reviews/vader.pkl", "rb") as f:
    vader = pickle.load(f)

print(reviews[0])
data = [review["text"] for review in reviews]
tokenized = tokenizer(data[:20], return_tensors='pt',padding="max_length", truncation=True)
print(len(data))
print(len(tokenized['input_ids']))
print(len(tokenized['input_ids'][0]))
#print(model(tokenized["input_ids"], attention_mask=tokenized["attention_mask"], token_type_ids=tokenized["token_type_ids"]))
"""
output = model(**tokenized)

print(output.__dict__)
"""