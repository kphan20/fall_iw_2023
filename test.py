from os.path import join
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from data import load_data, load_word2vec_data
import copy
from sklearn.metrics import confusion_matrix, classification_report
from model import MultiTaskDistilbert, TanModel

slurm_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 3))

def calc_hits(logits, labels):
    predictions = torch.argmax(torch.sigmoid(logits), 1)
    truths = torch.argmax(labels, 1)
    return torch.sum(predictions == truths).item()

def test(directory, model_class=None, hugging_face=False, is_word2vec=False, is_combined=False):
    if hugging_face:
        classifier = AutoModelForSequenceClassification.from_pretrained(directory)
    else:
        stuff = torch.load(join(directory, "checkpoint.pth"))
        classifier = model_class(**stuff["config"])
        classifier.load_state_dict(stuff["model"])
    

    _, _, test_dataset = load_word2vec_data(is_combined, use_sarcasm=True) if is_word2vec else load_data(None, use_sarcasm=not hugging_face)

    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier.to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    true_labels = []
    raw_predictions = []
    tan_masking = []
    double_masking = []

    tan_masking_true = []
    double_masking_true = []

    running_loss = 0
    running_corrects = 0

    classifier.eval()
    for batch in test_dataloader:
        batch_labels = torch.argmax(batch["labels"], 1)
        true_labels.append(copy.deepcopy(batch_labels))
        for key, value in batch.items():
            batch[key] = value.to(device)
        batch["input_ids"] = batch["input_ids"].squeeze()
        if hugging_face:
            outputs = classifier(**batch)
            running_loss += outputs.loss.item()
        else:
            outputs = classifier(**batch, has_sarcasm=True)
            running_loss += outputs.primary_loss
        running_corrects += calc_hits(outputs.logits, batch["labels"])
        raw = torch.argmax(torch.sigmoid(outputs.logits), 1)
        raw_predictions.append(copy.deepcopy(raw))
        if not hugging_face:
            sarcasm_labels = torch.argmax(batch["secondary_labels"], 1)
            sarcasm_pred = torch.argmax(torch.sigmoid(outputs.secondary_logits), 1)
            tan_mask_true = torch.logical_not(sarcasm_labels, out=torch.empty(sarcasm_labels.size(0), dtype=torch.int32, device=device))
            tan_mask_pred = torch.logical_not(sarcasm_pred, out=torch.empty(sarcasm_labels.size(0), dtype=torch.int32, device=device))
            tan_masking_true.append(torch.logical_and(raw, tan_mask_true, out=torch.empty(raw.size(0), dtype=torch.int32, device=device)))
            tan_masking.append(torch.logical_and(raw, tan_mask_pred, out=torch.empty(raw.size(0), dtype=torch.int32, device=device)))
            double_masking.append(torch.logical_xor(raw, sarcasm_pred, out=torch.empty(raw.size(0), dtype=torch.int32, device=device)))
            double_masking_true.append(torch.logical_xor(raw, sarcasm_labels, out=torch.empty(raw.size(0), dtype=torch.int32, device=device)))

    true_labels_concat = torch.cat(true_labels)
    print(f"Model: {directory}")
    print(f"Loss: {running_loss / len(test_dataloader)}")
    print(f"Acc: {running_corrects / (len(test_dataloader) * batch_size)}")
    raw_cat = torch.cat(raw_predictions).cpu()

    print(classification_report(true_labels_concat, raw_cat))

    print(confusion_matrix(true_labels_concat, raw_cat))

    if not hugging_face:
        tan_masking_cat = torch.cat(tan_masking).cpu()
        tan_masking_true_cat = torch.cat(tan_masking_true).cpu()
        double_masking_cat = torch.cat(double_masking).cpu()
        double_masking_true_cat = torch.cat(double_masking_true).cpu()
        print(classification_report(true_labels_concat, tan_masking_cat))
        print(classification_report(true_labels_concat, tan_masking_true_cat))
        print(classification_report(true_labels_concat, double_masking_cat))
        print(classification_report(true_labels_concat, double_masking_true_cat))
        print(confusion_matrix(true_labels_concat, tan_masking_cat))
        print(confusion_matrix(true_labels_concat, tan_masking_true_cat))
        print(confusion_matrix(true_labels_concat, double_masking_cat))
        print(confusion_matrix(true_labels_concat, double_masking_true_cat))
    

def test1():
    test("/scratch/network/kphan/COS_IW/models/tan_sentiment0_False", model_class=TanModel, is_combined=False)

def test2():
    test("/scratch/network/kphan/COS_IW/models/tan_multi3_False", model_class=TanModel, is_combined=False)

def test3():
    test("/scratch/network/kphan/COS_IW/models/tan_sequential00_False", model_class=TanModel, is_combined=True)

models = [test1, test2, test3]
models[slurm_idx]()

#test("/scratch/network/kphan/COS_IW/models/distilbert-base-uncased0newsplit", hugging_face=True)
#test("/scratch/network/kphan/COS_IW/models/distilbert_multi0", model_class=MultiTaskDistilbert)