import matplotlib.pyplot as plt
import pickle
import torch

with open("models/distilbert-base-uncased3/losses.pkl", "rb") as f:
    losses = pickle.load(f)

batch_size = losses["hp"]["batch_size"]
train_losses = [loss * batch_size / 15000 for loss in losses["train"]]
val_losses = [loss * batch_size/5000 for loss in losses["val"]]

plt.figure(0)

plt.plot(list(range(len(train_losses))), train_losses, label="Training loss")
plt.plot(list(range(len(val_losses))), val_losses, label="Validation loss")
plt.legend()
plt.title("Losses over training epochs")
plt.savefig("losses.png")

with open("models/distilbert-base-uncased2/losses.pkl", "rb") as f:
    losses = pickle.load(f)

batch_size = losses["hp"]["batch_size"]
train_losses = [loss * batch_size / 15000 for loss in losses["train"]]
val_losses = [loss * batch_size/5000 for loss in losses["val"]]

plt.figure(1)
plt.plot(list(range(len(train_losses))), train_losses, label="Training")
plt.plot(list(range(len(val_losses))), val_losses, label="Validation")
plt.legend()

plt.savefig("acc.png")