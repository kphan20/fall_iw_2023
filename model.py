import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

class GenericOutput:
    def __init__(self, loss):
        self.loss = loss

# https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/
class MultiTaskClassifier(nn.Module):
    def __init__(self, freeze_bert=True, from_pretrained=True, pretrained_path='distilbert-base-uncased'):
        if from_pretrained:
            self.bert_model = AutoModel.from_pretrained(pretrained_path)
        else:
            config = AutoConfig.from_pretrained(pretrained_path)
            self.bert_model = AutoModel.from_config(config)
        
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.freeze_embedding = freeze_bert

        dim = self.bert_model.config.dim # output size of model

        lstm_output_size = 20
        fc_dropout = 0.4
        self.lstm = nn.LSTM(dim, lstm_output_size/2, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(lstm_output_size, 180)
        self.relu = nn.ReLU()
        self.sentiment_output = nn.Linear(180, 2)
        self.sarcasm_output = nn.Linear(180, 2)
        self.dropout = nn.Dropout(fc_dropout)

        self.loss_fct = nn.BCEWithLogitsLoss()
    
    def forward(self, x, labels, is_sarcasm=False):
        # https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
        if self.freeze_bert:
            self.bert_model.eval()
        x = self.bert_model(**x)

        hidden_state = x[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        x, _ = self.lstm(pooled_output)
        x = self.dropout(self.relu(self.fc(x)))
        loss1 = self.loss_fct(self.sentiment_output(x), labels)
        loss2 = self.loss_fct(self.sarcasm_output(x), labels) #change this
        return GenericOutput(loss1 + loss2)


