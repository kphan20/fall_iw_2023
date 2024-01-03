from transformers import pipeline
import torch
import pickle

pipe = pipeline("text-classification", model="jkhan447/sarcasm-detection-Bert-base-uncased", 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(pipe("I’m soo excited to cook a three course Christmas dinner for my whole family this year!"))

with open("steam_reviews/reviews.pkl", "rb") as f:
    reviews = pickle.load(f)

for i, review in enumerate(reviews):
    output = pipe(reviews[i]["text"], **{'padding': True, "truncation": True, 'max_length': 512})
    if output[0]["label"] == "LABEL_1":
        print("Index:", i)
        print("Text:", reviews[i]["text"])
        print("Score:", reviews[i]["score"])