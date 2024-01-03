from torch.utils.data import Dataset, DataLoader
import csv
import pickle
import torch

class SteamDataset(Dataset):
    def __init__(self, reviews, scores, sarcasms):
        self.reviews = reviews
        self.scores = scores
        self.sarcasms = sarcasms
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = {"labels": torch.zeros(2)}
        result["labels"][self.scores[idx]] = 1
        #print(result)
        for key, value in self.reviews[idx].items():
            result[key] = value#value.to(self.device)
        return result
        #return self.reviews[idx].to(self.device), Tensor(self.scores[idx], device=self.device), Tensor(self.sarcasms[idx], device=self.device)

class SarcasmDataset(Dataset):
    def __init__(self, reviews, scores):
        self.reviews = reviews
        self.scores = scores
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = {"labels": self.scores[idx]}
        for key, value in self.reviews[idx].items():
            result[key] = value
        return result

def load_sarcasm_data(tokenizer, use_cached=True):
    if use_cached:
        with open("sarcasm/datasets.pkl", "rb") as f:
            datasets = pickle.load(f)
            train_data_tokens = datasets["train"]["tokens"]
            train_data_score = datasets["train"]["scores"]
            val_data_tokens = datasets["val"]["tokens"]
            val_data_score = datasets["val"]["scores"]
            test_data_tokens = datasets["test"]["tokens"]
            test_data_score = datasets["test"]["scores"]
    else:
        train_data_tokens = []
        train_data_score = []
        val_data_tokens = []
        val_data_score = []
        test_data_tokens = []
        test_data_score = []
        with open("sarcasm/trainEn.csv", "r", newline='', encoding="utf-8") as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text = tokenizer(row[1].strip(), return_tensors='pt',padding="max_length", truncation=True)
                train_data_tokens.append(text)
                score = torch.zeros(2)
                score[int(row[2].strip())] = 1
                train_data_score.append(score)
        with open("sarcasm/testEn.csv", "r", newline='', encoding="utf-8") as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text = tokenizer(row[0].strip(), return_tensors='pt',padding="max_length", truncation=True)
                test_data_tokens.append(text)
                score = torch.zeros(2)
                score[int(row[1].strip())] = 1
                test_data_score.append(score)
        with open("sarcasm/datasets.pkl", 'wb') as f:
            train = {"tokens": train_data_tokens, "scores": train_data_score}
            val = {"tokens": val_data_tokens, "scores": val_data_score}
            test = {"tokens": test_data_tokens, "scores": test_data_score}
            pickle.dump({"train": train, "val": val, "test": test}, f)

    return (SarcasmDataset(train_data_tokens, train_data_score), 
           SarcasmDataset(val_data_tokens, val_data_score), 
           SarcasmDataset(test_data_tokens, test_data_score))

def load_data(tokenizer, use_cached=True):
    if use_cached:
        with open("steam_reviews/datasets.pkl", "rb") as f:
            datasets = pickle.load(f)
            train_data_tokens = datasets["train"]["tokens"]
            train_data_score = datasets["train"]["scores"]
            train_data_sarcasm = datasets["train"]["sarcasm"]
            val_data_tokens = datasets["val"]["tokens"]
            val_data_score = datasets["val"]["scores"]
            val_data_sarcasm = datasets["val"]["sarcasm"]
            test_data_tokens = datasets["test"]["tokens"]
            test_data_score = datasets["test"]["scores"]
            test_data_sarcasm = datasets["test"]["sarcasm"]
    else:
        with open("steam_reviews/reviews.pkl", "rb") as f:
            reviews = pickle.load(f)
        with open("steam_reviews/neg_n.pkl", "rb") as f:
            neg_n = pickle.load(f)
        with open("steam_reviews/neg_y.pkl", "rb") as f:
            neg_y = pickle.load(f)
        with open("steam_reviews/pos_n.pkl", "rb") as f:
            pos_n = pickle.load(f)
        with open("steam_reviews/pos_y.pkl", "rb") as f:
            pos_y = pickle.load(f)
        
        train_data_tokens = []
        train_data_score = []
        train_data_sarcasm = []
        val_data_tokens = []
        val_data_score = []
        val_data_sarcasm = []
        test_data_tokens = []
        test_data_score = []
        test_data_sarcasm = []

        for idx, i in enumerate(random.sample(neg_n, 7500)):
            curr_review = reviews[i]
            curr_review["text"] = tokenizer(curr_review["text"],
                return_tensors='pt',padding="max_length", truncation=True)
            curr_review["sarcasm"] = 0
            curr_review["score"] = (curr_review["score"] + 1) >> 1
            if idx < 5000:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 6250:
                val_data_tokens.append(curr_review["text"])
                val_data_score.append(curr_review["score"])
                val_data_sarcasm.append(curr_review["sarcasm"])
            else:
                test_data_tokens.append(curr_review["text"])
                test_data_score.append(curr_review["score"])
                test_data_sarcasm.append(curr_review["sarcasm"])

        for idx, i in enumerate(random.sample(pos_n, 7500)):
            curr_review = reviews[i]
            curr_review["text"] = tokenizer(curr_review["text"],
                return_tensors='pt',padding="max_length", truncation=True)
            curr_review["sarcasm"] = 0
            curr_review["score"] = (curr_review["score"] + 1) >> 1
            if idx < 5000:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 6250:
                val_data_tokens.append(curr_review["text"])
                val_data_score.append(curr_review["score"])
                val_data_sarcasm.append(curr_review["sarcasm"])
            else:
                test_data_tokens.append(curr_review["text"])
                test_data_score.append(curr_review["score"])
                test_data_sarcasm.append(curr_review["sarcasm"])

        for idx, i in enumerate(random.sample(neg_y, 3750)):
            curr_review = reviews[i]
            curr_review["text"] = tokenizer(curr_review["text"],
                return_tensors='pt',padding="max_length", truncation=True)
            curr_review["sarcasm"] = 1
            curr_review["score"] = (curr_review["score"] + 1) >> 1
            if idx < 2500:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 3750:
                val_data_tokens.append(curr_review["text"])
                val_data_score.append(curr_review["score"])
                val_data_sarcasm.append(curr_review["sarcasm"])
            else:
                test_data_tokens.append(curr_review["text"])
                test_data_score.append(curr_review["score"])
                test_data_sarcasm.append(curr_review["sarcasm"])

        for idx, i in enumerate(random.sample(pos_y, 3750)):
            curr_review = reviews[i]
            curr_review["text"] = tokenizer(curr_review["text"],
                return_tensors='pt',padding="max_length", truncation=True)
            curr_review["sarcasm"] = 1
            curr_review["score"] = (curr_review["score"] + 1) >> 1
            if idx < 2500:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 3750:
                val_data_tokens.append(curr_review["text"])
                val_data_score.append(curr_review["score"])
                val_data_sarcasm.append(curr_review["sarcasm"])
            else:
                test_data_tokens.append(curr_review["text"])
                test_data_score.append(curr_review["score"])
                test_data_sarcasm.append(curr_review["sarcasm"])
        
        with open("steam_reviews/datasets.pkl", 'wb') as f:
            train = {"tokens": train_data_tokens, "scores": train_data_score, "sarcasm": train_data_sarcasm}
            val = {"tokens": val_data_tokens, "scores": val_data_score, "sarcasm": val_data_sarcasm}
            test = {"tokens": test_data_tokens, "scores": test_data_score, "sarcasm": test_data_sarcasm}
            pickle.dump({"train": train, "val": val, "test": test}, f)
    
    return (SteamDataset(train_data_tokens, train_data_score, train_data_sarcasm), 
           SteamDataset(val_data_tokens, val_data_score, val_data_sarcasm), 
           SteamDataset(test_data_tokens, test_data_score, test_data_sarcasm))