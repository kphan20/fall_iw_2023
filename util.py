from pathlib import Path
import csv
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize

stop_words = set(stopwords.words("english"))
def parallel_tokenize(review):
    words = word_tokenize(review)
    return [word for word in words if word.casefold() not in stop_words]

def read_steam_reviews(use_cache=True, update_cache=False):
    reviews = []
    
    if use_cache and Path("steam_reviews/reviews.pkl").is_file():
        with open("steam_reviews/reviews.pkl", "rb") as f:
            return pickle.load(f)

    with open("steam_reviews/dataset.csv", "r", encoding="utf-8") as f:
        f.readline() # app_id, app_name, review_text, review_score, review_votes
        csv_reader = csv.reader(f)
        for row in csv_reader:
            review_dict = {}
            review_dict["text"] = row[2].rstrip()
            review_dict["score"] = int(row[3].strip())
            reviews.append(review_dict)
    
    if update_cache:
        with open("steam_reviews/reviews.pkl", "wb") as f:
            pickle.dump(reviews, f)
            
    return reviews