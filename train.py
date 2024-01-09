from torch.utils.data import DataLoader
import random
import pickle
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.optim import Adam, RMSprop
from torch import cuda, Tensor, manual_seed
from tqdm.auto import tqdm
from torch.nn.functional import one_hot
from data import load_data, load_word2vec_data, load_word2vec_sarcasm_data, load_sarcasm_data
import os
from pathlib import Path
import matplotlib.pyplot as plt
from model import MultiTaskDistilbert, TanModel

manual_seed(497)

batch_sizes = [16, 16, 16, 16]
lrs = [0.00001, 0.00001, 0.00001, 0.00001]#[0.000001, 0.000005, 0.00001, 0.00002]#[0.00002,0.00002,0.00002,0.00002]#[0.000001, 0.000005, 0.00001, 0.00002]
#lrs = [0.0001, 0.0005, 0.001, 0.002]#[0.0001, 0.0001, 0.0001, 0.0001]
dropouts = [0.3, 0.4, 0.5, 0.6]

# exp 1 - lr=0.0001, dropout=0.2, batch_size=16, epochs = 10
# exp 2 - lr=0.00002, dropout=0.2, batch_size=16, epochs = 15
# exp 3 - lr=0.00001, dropout=0.3, batch_size=16, epochs = 15

slurm_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 3))

def calc_hits(logits, labels):
    predictions = torch.argmax(torch.sigmoid(logits), 1)
    truths = torch.argmax(labels, 1)
    return torch.sum(predictions==truths).item()

def train_model(model_name, classifier, dataloaders, directory="/scratch/network/kphan/COS_IW/models", use_scheduler=True, has_sentiment=True, has_sarcasm=False, hugging_face=False, epochs=10, is_word2vec=False):
    result_path = os.path.join(directory, model_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # HYPERPARAMETERS
    # https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/distilbert#transformers.DistilBertConfig
    batch_size = batch_sizes[slurm_idx]
    num_epochs = epochs
    lr = lrs[slurm_idx]

    device = torch.device("cuda" if cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    optimizer = RMSprop(classifier.parameters(), lr=lr) if is_word2vec else Adam(classifier.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_dataloader)

    if use_scheduler:
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
            num_warmup_steps=0, num_training_steps=num_training_steps)

    classifier.to(device)

    losses = []
    val_losses = []
    hits = []
    val_hits = []

    prim_losses = []
    val_prim_losses = []
    sec_losses = []
    val_sec_losses = []

    sec_hits = []
    val_sec_hits = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_prim_loss = 0
        epoch_sec_loss = 0
        
        epoch_hits = 0
        epoch_sec_hits = 0
        
        num_train_batches = 0
        num_val_batches = 0
        for batch in train_dataloader:
            classifier.train()
            for key, value in batch.items():
                batch[key] = value.to(device)
            batch["input_ids"] = batch["input_ids"].squeeze()
            if hugging_face:
                outputs = classifier(**batch)
            else:
                outputs = classifier(**batch, has_sentiment=has_sentiment, has_sarcasm=has_sarcasm)
                if outputs.secondary_logits is not None:
                    epoch_sec_hits += calc_hits(outputs.secondary_logits, batch["secondary_labels"])
                if outputs.primary_loss is not None:
                    epoch_prim_loss += outputs.primary_loss
                if outputs.secondary_loss is not None:
                    epoch_sec_loss += outputs.secondary_loss

            epoch_hits += calc_hits(outputs.logits, batch["labels"])
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                lr_scheduler.step()
            optimizer.zero_grad()
            num_train_batches += 1

        val_loss = 0
        val_prim_loss = 0
        val_sec_loss = 0

        val_hit = 0
        val_sec_hit = 0
        for batch in val_dataloader:
            classifier.eval()
            for key, value in batch.items():
                batch[key] = value.to(device)
            batch["input_ids"] = batch["input_ids"].squeeze()
            if hugging_face:
                outputs = classifier(**batch)
            else:
                outputs = classifier(**batch, has_sentiment=has_sentiment, has_sarcasm=has_sarcasm)
                if outputs.secondary_logits is not None:
                    val_sec_hit += calc_hits(outputs.secondary_logits, batch["secondary_labels"])
                if outputs.primary_loss is not None:
                    val_prim_loss += outputs.primary_loss
                if outputs.secondary_loss is not None:
                    val_sec_loss += outputs.secondary_loss
            val_hit += calc_hits(outputs.logits, batch["labels"])
            loss = torch.sum(outputs.loss)
            val_loss += loss.item()
            num_val_batches += 1
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        hits.append(epoch_hits)
        val_hits.append(val_hit)

        prim_losses.append(epoch_prim_loss)
        sec_losses.append(epoch_sec_loss)
        sec_hits.append(epoch_sec_hits)

        val_prim_losses.append(val_prim_loss)
        val_sec_losses.append(val_sec_loss)
        val_sec_hits.append(val_sec_hit)

        print(losses)
        print(val_losses)
        print(hits)
        print(val_hits)
        if hugging_face:
            classifier.save_pretrained(result_path, from_pt=True)
        else:
            torch.save({"model": classifier.state_dict(), "config": classifier.config_dict}, os.path.join(result_path, "checkpoint.pth"))
        print(f"Finished epoch {epoch}")

    with open(os.path.join(result_path, "losses.pkl"), "wb") as f:
        hyperparams = {"batch_size": batch_size, "num_epochs": num_epochs, "lr":lr, "dropout": dropouts[slurm_idx]}
        stuff = {"train":losses,"val":val_losses, "train_acc":hits, "val_acc":val_hits, "hp": hyperparams}
        stuff["train_extra"] = {"prim": prim_losses, "sec": sec_losses, "sec_hits": sec_hits}
        stuff["val_extra"] = {"prim": val_prim_losses, "sec": val_sec_losses, "sec_hits": val_sec_hits}
        pickle.dump(stuff, f)

    train_losses = [loss / num_train_batches for loss in losses]
    train_acc = [hit / (len(train_dataloader) * batch_size) for hit in hits]

    val_losses = [loss / num_val_batches for loss in val_losses]
    val_acc = [hit / (len(val_dataloader) * batch_size) for hit in val_hits]

    plt.figure(0)
    plt.plot(list(range(len(train_losses))), train_losses, label="Training loss")
    plt.plot(list(range(len(val_losses))), val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses per batch over training epochs")

    plt.savefig(os.path.join(result_path, "losses.png"))

    plt.close()
    plt.figure(1)
    plt.plot(list(range(len(train_acc))), train_acc, label="Training accuracy")
    plt.plot(list(range(len(val_acc))), val_acc, label="Validation accuracy")
    plt.legend()
    plt.title("Overall accuracy over training epochs")

    plt.savefig(os.path.join(result_path, "acc.png"))

    plt.close()
def train_distilbert(use_scheduler):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, seq_classif_dropout=dropouts[slurm_idx])
    
    # only fine tuning last couple of layers
    for param in classifier.distilbert.parameters():
        param.requires_grad = False

    batch_size = batch_sizes[slurm_idx]

    train_dataset, val_dataset, test_dataset = load_data(tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    train_model(f"{model_name}{slurm_idx}newsplit_{use_scheduler}", classifier, (train_dataloader, val_dataloader, test_dataloader), use_scheduler=use_scheduler, hugging_face=True)

def train_distilbert_multi():
    model_name = "distilbert_multi"
    classifier = MultiTaskDistilbert(dropouts[slurm_idx])
    
    # only fine tuning last couple of layers
    for param in classifier.distilbert.parameters():
        param.requires_grad = False

    batch_size = batch_sizes[slurm_idx]

    train_dataset, val_dataset, test_dataset = load_data(None, use_sarcasm=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    train_model(f"{model_name}{slurm_idx}", classifier, (train_dataloader, val_dataloader, test_dataloader), has_sentiment=True, has_sarcasm=True)

def train_distilbert_scratch():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, dropout=dropouts[slurm_idx], num_labels=2)
    classifier = AutoModelForSequenceClassification.from_config(config)

    batch_size = batch_sizes[slurm_idx]

    train_dataset, val_dataset, test_dataset = load_data(tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    train_model(f"{model_name}-scratchsd{slurm_idx}", classifier, (train_dataloader, val_dataloader, test_dataloader))

def train_tan(model_name="tan_sentiment", is_word2vec=True, is_combined=False, use_sarcasm=False):
    classifier = TanModel(is_word2vec, dropout=dropouts[slurm_idx], is_combined=is_combined)
    print(dropouts[slurm_idx])
    batch_size = batch_sizes[slurm_idx]

    train_dataset, val_dataset, test_dataset = load_word2vec_data(is_combined, use_sarcasm=use_sarcasm) if is_word2vec else load_data(None, use_sarcasm=use_sarcasm)
    print(len(train_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    train_model(f"{model_name}{slurm_idx}_{is_word2vec}", classifier, (train_dataloader, val_dataloader, test_dataloader), has_sentiment=True, has_sarcasm=use_sarcasm, is_word2vec=is_word2vec, epochs=15)

def train_tan_sentiment(is_word2vec):
    train_tan(is_word2vec=is_word2vec, is_combined=False, use_sarcasm=False)

def train_tan_multi(is_word2vec):
    train_tan("tan_multi", is_word2vec, is_combined=False, use_sarcasm=True)

def train_tan_sequential(is_word2vec):
    classifier = TanModel(is_word2vec, dropout=dropouts[slurm_idx], is_combined=True)

    batch_size = batch_sizes[slurm_idx]

    train_dataset, val_dataset = load_word2vec_sarcasm_data() if is_word2vec else load_sarcasm_data(None)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    train_model(f"tan_intermediate{slurm_idx}_{is_word2vec}", classifier, (train_dataloader, val_dataloader, None), has_sentiment=False, has_sarcasm=True, epochs=5)
    train_tan(f"tan_sequential{slurm_idx}", is_word2vec, is_combined=True, use_sarcasm=False)

# experiment 1
# train_distilbert(True)
#train_distilbert_multi()

# experiment 2
#train_tan_sentiment(True)
"""
first_half = slurm_idx < 4
slurm_idx = slurm_idx % 4
if first_half:
    train_tan_multi(True)
else:
    train_tan_sequential(True)
"""

# experiment 3
#train_tan_sentiment(False)
#train_tan_multi(False)
#train_tan_sequential(False)

first_half = slurm_idx < 4
slurm_idx = slurm_idx % 4
if first_half:
    train_tan_multi(False)
else:
    train_tan_sequential(False)