import matplotlib.pyplot as plt
import pickle
import torch

with open("models/distilbert-base-uncased3/losses.pkl", "rb") as f:
    losses = pickle.load(f)

batch_size = losses["hp"]["batch_size"]
train_losses = [loss * batch_size / 15000 for loss in losses["train"]]
val_losses = [loss * batch_size/5000 for loss in losses["val"]]

plt.plot(list(range(len(train_losses))), train_losses)
plt.plot(list(range(len(val_losses))), val_losses)
plt.savefig("test8.png")