from torch.utils.data import DataLoader
import random
import pickle
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.optim import Adam
from torch import cuda, Tensor, manual_seed
from tqdm.auto import tqdm
from torch.nn.functional import one_hot
from data import load_data
import os
from pathlib import Path

manual_seed(497)

batch_sizes = [16, 16, 16, 16]
#lrs = [0.000001, 0.000005, 0.00001, 0.00002]
lrs = [0.0001, 0.0005, 0.001, 0.002]
dropouts = [0.2, 0.5, 0.5, 0.5]

slurm_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 3))

def calc_hits(logits, labels):
    predictions = torch.argmax(torch.sigmoid(logits), 1)
    truths = torch.argmax(labels, 1)
    return torch.sum(predictions*truths).item()

def train_model(model_name, classifier, dataloaders, directory="/scratch/network/kphan/COS_IW/models", use_scheduler=True):
    result_path = os.path.join(directory, model_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # HYPERPARAMETERS
    # https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/distilbert#transformers.DistilBertConfig
    batch_size = batch_sizes[slurm_idx]
    num_epochs = 10
    lr = lrs[slurm_idx]

    device = torch.device("cuda" if cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    optimizer = Adam(classifier.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_dataloader)

    if use_scheduler:
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
            num_warmup_steps=0, num_training_steps=num_training_steps)

    classifier.to(device)

    losses = []
    val_losses = []
    hits = []
    val_hits = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_hits = 0
        for batch in train_dataloader:
            classifier.train()
            for key, value in batch.items():
                batch[key] = value.to(device)
            batch["input_ids"] = batch["input_ids"].squeeze()
            outputs = classifier(**batch)
            epoch_hits += calc_hits(outputs.logits, batch["labels"])
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                lr_scheduler.step()
            optimizer.zero_grad()

        val_loss = 0
        val_hit = 0
        for batch in val_dataloader:
            classifier.eval()
            for key, value in batch.items():
                batch[key] = value.to(device)
            batch["input_ids"] = batch["input_ids"].squeeze()
            outputs = classifier(**batch)
            val_hit += calc_hits(outputs.logits, batch["labels"])
            loss = torch.sum(outputs.loss)
            val_loss += loss.item()
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        hits.append(epoch_hits)
        val_hits.append(val_hit)
        print(losses)
        print(val_losses)
        print(hits)
        print(val_hits)
        classifier.save_pretrained(result_path, from_pt=True)
        print(f"Finished epoch {epoch}")

    with open(os.path.join(result_path, "losses.pkl"), "wb") as f:
        hyperparams = {"batch_size": batch_size, "num_epochs": num_epochs, "lr":lr, "dropout": dropouts[slurm_idx]}
        pickle.dump({"train":losses,"val":val_losses, "train_acc":hits, "val_acc":val_hits, "hp": hyperparams}, f)

def train_distilbert():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # only fine tuning last couple of layers
    for param in classifier.distilbert.parameters():
        param.requires_grad = False

    batch_size = batch_sizes[slurm_idx]

    train_dataset, val_dataset, test_dataset = load_data(tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    train_model(f"{model_name}{slurm_idx}", classifier, (train_dataloader, val_dataloader, test_dataloader))

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

def train_distilbert_sarcasm():
    """
    classifier = MultiTaskClassifier()
    train_dataset, val_dataset, test_dataset = load_data(tokenizer)
    train_dataset_s, val_dataset_s, test_dataset_s = load_sarcasm_data(tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    """
    pass

train_distilbert()
# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # load_data(tokenizer)
# device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
# train_dataset, val_dataset, test_dataset = load_data(tokenizer)
# batch_size = 32
# num_epochs = 10

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)#MultiTaskClassifier()

# optimizer = Adam(classifier.parameters(), lr=0.0001)
# #loss_fn = nn.BCELoss()

# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
#     num_warmup_steps=0, num_training_steps=num_training_steps)

# classifier.to(device)

# progress_bar = tqdm(range(num_training_steps))

# losses = []
# val_losses = []

# for epoch in range(num_epochs):
#     epoch_loss = 0
#     for batch in train_dataloader:
#         classifier.train()
#         for key, value in batch.items():
#             batch[key] = value.to(device)
#         batch["input_ids"] = batch["input_ids"].squeeze()
#         outputs = classifier(**batch)
#         loss = outputs.loss
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
#         #y_pred = classifier(**batch)
#         #loss = loss_fn(y_pred, )
#         #classifier.eval()
#     val_loss = 0
#     for batch in val_dataloader:
#         classifier.eval()
#         for key, value in batch.items():
#             batch[key] = value.to(device)
#         batch["input_ids"] = batch["input_ids"].squeeze()
#         outputs = classifier(**batch)
#         loss = torch.sum(outputs.loss)
#         val_loss += loss.item()
#     losses.append(epoch_loss)
#     val_losses.append(val_loss)
#     classifier.save_pretrained("/scratch/network/kphan/COS_IW/models/baseline", from_pt=True)
#     print(f"Finished epoch {epoch}")

# with open("losses.pkl", "wb") as f:
#     pickle.dump({"train":losses,"val":val_losses}, f)
