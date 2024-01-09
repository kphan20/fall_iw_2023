import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from torchtext.vocab import Vectors

class GenericOutput:
    def __init__(self, loss, logits, secondary_logits=None, primary_loss=None, secondary_loss=None):
        self.loss = loss
        self.logits = logits
        self.secondary_logits = secondary_logits
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        

class Word2VecEmbedding(nn.Module):
    def __init__(self, is_combined, output_size = 300):
        super().__init__()
        word2vec_file = "combined.txt" if is_combined else "test.txt"
        vectors = Vectors(name=word2vec_file)
        self.embed = nn.Embedding.from_pretrained(torch.cat([vectors.vectors, torch.zeros((1, 300))]))
        self.output_size = output_size
    
    def forward(self, input_ids=None, attention_mask=None):
        output = self.embed(input_ids)
        return [output]


class MultiTaskDistilbert(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.config_dict = {}
        self.distilbert = AutoModel.from_pretrained("distilbert-base-uncased")

        dim = self.distilbert.config.dim
        # only fine tuning last couple of layers
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        self.pre_classifier = nn.Linear(dim, dim)
        self.sentiment_output = nn.Linear(dim, 2)
        self.sarcasm_output = nn.Linear(dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.loss_fct = nn.BCEWithLogitsLoss()
        
    
    def forward(self, input_ids, labels, attention_mask = None, has_sentiment=True, has_sarcasm=False, secondary_labels=None):
        # https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        x = self.dropout(pooled_output)  # (bs, dim)

        if has_sentiment and has_sarcasm: # multi-task
            sentiment_logits = self.sentiment_output(x)
            sarcasm_logits = self.sarcasm_output(x)
            loss1 = self.loss_fct(sentiment_logits, labels)
            loss2 = self.loss_fct(sarcasm_logits, secondary_labels)
            return GenericOutput(loss1 + loss2, sentiment_logits, sarcasm_logits, loss1.item(), loss2.item())
        elif has_sentiment: # sentiment only
            sentiment_logits = self.sentiment_output(x)
            loss = self.loss_fct(sentiment_logits, labels)
            return GenericOutput(loss, sentiment_logits)

class TanModel(nn.Module):
    def __init__(self, is_word2vec, lstm_output_size=64, dropout=0.2, is_combined=False):
        super().__init__()
        # save these variables for model reloading
        self.config_dict = {"is_combined": is_combined, "is_word2vec": is_word2vec, "lstm_output_size": lstm_output_size}

        if is_word2vec:
            self.embed = Word2VecEmbedding(is_combined)
            dim = self.embed.output_size
            seq_length = 75
        else:
            self.embed = AutoModel.from_pretrained("distilbert-base-uncased")

            for param in self.embed.parameters():
                param.requires_grad = False

            dim = self.embed.config.dim # output size of model
            seq_length = 512

        self.lstm = nn.LSTM(dim, lstm_output_size, num_layers=2, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(seq_length * lstm_output_size * 2, 180)
        self.relu = nn.ReLU()
        self.sentiment_output = nn.Linear(180, 2)
        self.sarcasm_output = nn.Linear(180, 2)
        self.dropout = nn.Dropout(dropout)

        self.loss_fct = nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids, labels, attention_mask = None, has_sentiment=True, has_sarcasm=False, secondary_labels=None):
        # https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
        self.embed.eval()
        x = self.embed(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = x[0]  # (bs, seq_len, dim)

        x, _ = self.lstm(hidden_state)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc(x)))

        if has_sentiment and has_sarcasm: # multi-task
            sentiment_logits = self.sentiment_output(x)
            sarcasm_logits = self.sarcasm_output(x)
            loss1 = self.loss_fct(sentiment_logits, labels)
            loss2 = self.loss_fct(sarcasm_logits, secondary_labels)
            return GenericOutput(loss1 + loss2, sentiment_logits, sarcasm_logits, loss1.item(), loss2.item())
        elif has_sentiment: # sentiment only
            sentiment_logits = self.sentiment_output(x)
            loss = self.loss_fct(sentiment_logits, labels)
            return GenericOutput(loss, sentiment_logits)
        elif has_sarcasm: # sarcasm only
            sarcasm_logits = self.sarcasm_output(x)
            loss = self.loss_fct(sarcasm_logits, labels)
            return GenericOutput(loss, sarcasm_logits)
