from torch.utils.data import Dataset, DataLoader
import csv
import pickle
import torch
import random
import torch.nn.functional as F

def resize_word2vec(output, output_size = 75, pad_idx=13035):
    return F.pad(output, (0, output_size - output.size(0)), "constant", pad_idx)

random.seed(479)
class SteamDataset(Dataset):
    def __init__(self, reviews, scores, sarcasms, use_sarcasm=False, is_word2vec=False, is_combined=False):
        self.reviews = reviews
        self.scores = scores
        self.sarcasms = sarcasms
        self.is_word2vec = is_word2vec
        self.use_sarcasm = use_sarcasm
        self.is_combined = is_combined
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = {"labels": torch.zeros(2)}
        result["labels"][self.scores[idx]] = 1
        #print(result)
        if self.is_word2vec:
            pad_idx = 13035 if self.is_combined else 11819
            result["input_ids"] = resize_word2vec(self.reviews[idx], pad_idx=pad_idx)
        else:
            for key, value in self.reviews[idx].items():
                result[key] = value#value.to(self.device)
        if self.use_sarcasm:
            result["secondary_labels"] = torch.zeros(2)
            result["secondary_labels"][self.sarcasms[idx]] = 1
        return result
        #return self.reviews[idx].to(self.device), Tensor(self.scores[idx], device=self.device), Tensor(self.sarcasms[idx], device=self.device)

class SarcasmDataset(Dataset):
    def __init__(self, reviews, scores, is_word2vec=False):
        self.reviews = reviews
        self.scores = scores
        self.is_word2vec = is_word2vec
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = {"labels": self.scores[idx]}

        if self.is_word2vec:
            result["input_ids"] = resize_word2vec(self.reviews[idx])
        else:
            for key, value in self.reviews[idx].items():
                result[key] = value

        return result


def load_sarcasm_data(tokenizer, use_cached=True):
    if use_cached:
        with open("sarcasm/datasets.pkl", "rb") as f:
            datasets = pickle.load(f)
            train_data_tokens = datasets["train"]["tokens"]
            train_data_score = datasets["train"]["scores"]
            test_data_tokens = datasets["test"]["tokens"]
            test_data_score = datasets["test"]["scores"]
    else:
        train_data_tokens = []
        train_data_score = []
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
            test = {"tokens": test_data_tokens, "scores": test_data_score}
            pickle.dump({"train": train, "test": test}, f)

    return (SarcasmDataset(train_data_tokens, train_data_score), 
           SarcasmDataset(test_data_tokens, test_data_score))

def load_data(tokenizer, use_sarcasm=False, use_cached=True):
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
            if idx < 6000:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 6750:
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
            if idx < 6000:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 6750:
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
            if idx < 3000:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 3375:
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
            if idx < 3000:
                train_data_tokens.append(curr_review["text"])
                train_data_score.append(curr_review["score"])
                train_data_sarcasm.append(curr_review["sarcasm"])
            elif idx < 3375:
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
    
    return (SteamDataset(train_data_tokens, train_data_score, train_data_sarcasm, use_sarcasm), 
           SteamDataset(val_data_tokens, val_data_score, val_data_sarcasm, use_sarcasm), 
           SteamDataset(test_data_tokens, test_data_score, test_data_sarcasm, use_sarcasm))

def load_word2vec_sarcasm_data():
    with open("sarcasm/word2vec_datasets2.pkl", "rb") as f:
        datasets = pickle.load(f)
        train_data_tokens = datasets["train"]["tokens"]
        train_data_score = datasets["train"]["scores"]
        test_data_tokens = datasets["test"]["tokens"]
        test_data_score = datasets["test"]["scores"]
    
    return (SarcasmDataset(train_data_tokens, train_data_score, True), 
            SarcasmDataset(test_data_tokens, test_data_score, True))

def load_word2vec_data(is_combined, use_sarcasm=False, use_cached=True):
    file_num = 3 if is_combined else 2
    with open(f"steam_reviews/word2vec_datasets{file_num}.pkl", "rb") as f:
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
    
    return (SteamDataset(train_data_tokens, train_data_score, train_data_sarcasm, use_sarcasm, True, is_combined=is_combined), 
           SteamDataset(val_data_tokens, val_data_score, val_data_sarcasm, use_sarcasm, True, is_combined=is_combined), 
           SteamDataset(test_data_tokens, test_data_score, test_data_sarcasm, use_sarcasm, True, is_combined=is_combined))

def word2vec_setup():
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    from gensim.models import Word2Vec

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    def preprocess(review):
        words = nltk.wordpunct_tokenize(review)
        # removes punctuation and stopwords, stems the words as well
        filtered = [stemmer.stem(word.lower()) for word in words if word.isalpha() and word.casefold() not in stop_words]
        return filtered
    
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
        
    sentences = []

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
        curr_review["text"] = preprocess(curr_review["text"])
        sentences.append(curr_review["text"])
        curr_review["sarcasm"] = 0
        curr_review["score"] = (curr_review["score"] + 1) >> 1
        if idx < 6000:
            train_data_tokens.append(curr_review["text"])
            train_data_score.append(curr_review["score"])
            train_data_sarcasm.append(curr_review["sarcasm"])
        elif idx < 6750:
            val_data_tokens.append(curr_review["text"])
            val_data_score.append(curr_review["score"])
            val_data_sarcasm.append(curr_review["sarcasm"])
        else:
            test_data_tokens.append(curr_review["text"])
            test_data_score.append(curr_review["score"])
            test_data_sarcasm.append(curr_review["sarcasm"])

    for idx, i in enumerate(random.sample(pos_n, 7500)):
        curr_review = reviews[i]
        curr_review["text"] = preprocess(curr_review["text"])
        sentences.append(curr_review["text"])
        curr_review["sarcasm"] = 0
        curr_review["score"] = (curr_review["score"] + 1) >> 1
        if idx < 6000:
            train_data_tokens.append(curr_review["text"])
            train_data_score.append(curr_review["score"])
            train_data_sarcasm.append(curr_review["sarcasm"])
        elif idx < 6750:
            val_data_tokens.append(curr_review["text"])
            val_data_score.append(curr_review["score"])
            val_data_sarcasm.append(curr_review["sarcasm"])
        else:
            test_data_tokens.append(curr_review["text"])
            test_data_score.append(curr_review["score"])
            test_data_sarcasm.append(curr_review["sarcasm"])

    for idx, i in enumerate(random.sample(neg_y, 3750)):
        curr_review = reviews[i]
        curr_review["text"] = preprocess(curr_review["text"])
        sentences.append(curr_review["text"])
        curr_review["sarcasm"] = 1
        curr_review["score"] = (curr_review["score"] + 1) >> 1
        if idx < 3000:
            train_data_tokens.append(curr_review["text"])
            train_data_score.append(curr_review["score"])
            train_data_sarcasm.append(curr_review["sarcasm"])
        elif idx < 3375:
            val_data_tokens.append(curr_review["text"])
            val_data_score.append(curr_review["score"])
            val_data_sarcasm.append(curr_review["sarcasm"])
        else:
            test_data_tokens.append(curr_review["text"])
            test_data_score.append(curr_review["score"])
            test_data_sarcasm.append(curr_review["sarcasm"])

    for idx, i in enumerate(random.sample(pos_y, 3750)):
        curr_review = reviews[i]
        curr_review["text"] = preprocess(curr_review["text"])
        sentences.append(curr_review["text"])
        curr_review["sarcasm"] = 1
        curr_review["score"] = (curr_review["score"] + 1) >> 1
        if idx < 3000:
            train_data_tokens.append(curr_review["text"])
            train_data_score.append(curr_review["score"])
            train_data_sarcasm.append(curr_review["sarcasm"])
        elif idx < 3375:
            val_data_tokens.append(curr_review["text"])
            val_data_score.append(curr_review["score"])
            val_data_sarcasm.append(curr_review["sarcasm"])
        else:
            test_data_tokens.append(curr_review["text"])
            test_data_score.append(curr_review["score"])
            test_data_sarcasm.append(curr_review["sarcasm"])
    
    with open("steam_reviews/word2vec_datasets.pkl", 'wb') as f:
        train = {"tokens": train_data_tokens, "scores": train_data_score, "sarcasm": train_data_sarcasm}
        val = {"tokens": val_data_tokens, "scores": val_data_score, "sarcasm": val_data_sarcasm}
        test = {"tokens": test_data_tokens, "scores": test_data_score, "sarcasm": test_data_sarcasm}
        pickle.dump({"train": train, "val": val, "test": test}, f)
    
    model = Word2Vec(sentences, min_count=2, vector_size=300, epochs=100, workers=4)
    model.wv.save("steam_reviews/steam.txt")

def combine_word2vec():
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.snowball import SnowballStemmer
    from gensim.models import Word2Vec

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    def preprocess(review):
        words = nltk.wordpunct_tokenize(review)
        # removes punctuation and stopwords, stems the words as well
        filtered = [stemmer.stem(word.lower()) for word in words if word.isalpha() and word.casefold() not in stop_words]
        return filtered
    
    train_data_tokens = []
    train_data_score = []
    test_data_tokens = []
    test_data_score = []

    sentences = []
    with open("sarcasm/trainEn.csv", "r", newline='', encoding="utf-8") as f:
        f.readline()
        csv_reader = csv.reader(f)
        for row in csv_reader:
            text = preprocess(row[1].strip())
            train_data_tokens.append(text)
            sentences.append(text)
            score = torch.zeros(2)
            score[int(row[2].strip())] = 1
            train_data_score.append(score)
    with open("sarcasm/testEn.csv", "r", newline='', encoding="utf-8") as f:
        f.readline()
        csv_reader = csv.reader(f)
        for row in csv_reader:
            text = preprocess(row[0].strip())
            sentences.append(text)
            test_data_tokens.append(text)
            score = torch.zeros(2)
            score[int(row[1].strip())] = 1
            test_data_score.append(score)
    with open("sarcasm/word2vec_datasets.pkl", 'wb') as f:
        train = {"tokens": train_data_tokens, "scores": train_data_score}
        test = {"tokens": test_data_tokens, "scores": test_data_score}
        pickle.dump({"train": train, "test": test}, f)
    
    with open("steam_reviews/word2vec_datasets.pkl", 'rb') as f:
        steam_datasets = pickle.load(f)
    
    sentences += steam_datasets["train"]["tokens"]
    sentences += steam_datasets["val"]["tokens"]
    sentences += steam_datasets["test"]["tokens"]

    model = Word2Vec(sentences, min_count=2, vector_size=300, epochs=100, workers=4)
    model.wv.save_word2vec_format("combined.txt")

def fix_tokens():
    from torchtext.vocab import Vectors
    import torch
    vectors = Vectors(name="combined.txt")

    with open("steam_reviews/word2vec_datasets.pkl", "rb") as f:
        datasets = pickle.load(f)
    
    with open("sarcasm/word2vec_datasets.pkl", "rb") as f:
        s_datasets = pickle.load(f)
    
    train_data_tokens = datasets["train"]["tokens"]
    val_data_tokens = datasets["val"]["tokens"]
    test_data_tokens = datasets["test"]["tokens"]
    
    train_tokens = s_datasets["train"]["tokens"]
    test_tokens = s_datasets["test"]["tokens"]
    combo = [train_data_tokens, val_data_tokens, test_data_tokens, train_tokens, test_tokens]

    new_combo = []
    for tokens in combo: # each dataset
        curr_tokens = []
        empty = 0
        for i, sentence in enumerate(tokens): # each sentence
            curr_sent = []
            for token in sentence: # each word
                idx = vectors.stoi.get(token, None)
                if idx is None:
                    continue
                curr_sent.append(idx)
            if len(curr_sent) == 0:
                empty += 1
            curr_tokens.append(torch.IntTensor(curr_sent))
        print(empty)
        new_combo.append(curr_tokens)
    
    datasets["train"]["tokens"] = new_combo[0]
    datasets["val"]["tokens"] = new_combo[1]
    datasets["test"]["tokens"] = new_combo[2]
    s_datasets["train"]["tokens"] = new_combo[3]
    s_datasets["test"]["tokens"] = new_combo[4]

    with open("steam_reviews/word2vec_datasets3.pkl", "wb") as f:
        pickle.dump(datasets, f)

    with open("sarcasm/word2vec_datasets2.pkl", "wb") as f:
        pickle.dump(s_datasets, f)
