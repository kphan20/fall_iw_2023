import csv
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer, word_tokenize
import nltk
#nltk.download('punkt')
import numpy as np
import re
import pickle

#tokenizer = RegexpTokenizer(r'\w+')
#stop_words = set(stopwords.words("english"))
#filtered_review_lengths = []

pos_sarcastic_review_lengths = []
pos_none_review_lengths = []
#pos_filtered_review_lengths = []
neg_sarcastic_review_lengths = []
neg_none_review_lengths = []
#neg_filtered_review_lengths = []

ignore_nonascii = re.compile(r'[^\x00-\x7f]')

game_counts = {}

with open("steam_reviews/vader.pkl", "rb") as f:
    vader = pickle.load(f)

# Load steam reviews text and rating
with open("steam_reviews/reviews.pkl", "rb") as f:
    reviews = pickle.load(f)

# Load Vader classification (supposed sarcasm)
with open("steam_reviews/neg_n.pkl", "rb") as f:
    neg_n = pickle.load(f)

with open("steam_reviews/neg_y.pkl", "rb") as f:
    neg_y = pickle.load(f)

with open("steam_reviews/pos_n.pkl", "rb") as f:
    pos_n = pickle.load(f)

with open("steam_reviews/pos_y.pkl", "rb") as f:
    pos_y = pickle.load(f)

raw_review_lengths = np.zeros(len(reviews))

mapping = np.zeros((len(raw_review_lengths), 2))

for idx, i in enumerate(neg_n):
    review = reviews[i]["text"].rstrip()
    words = word_tokenize(review)
    unfiltered_length = len(words)
    raw_review_lengths[i] = unfiltered_length
    neg_none_review_lengths.append(unfiltered_length)
    mapping[i][0] = 0
    mapping[i][1] = idx

for idx, i in enumerate(neg_y):
    review = reviews[i]["text"].rstrip()
    words = word_tokenize(review)
    unfiltered_length = len(words)
    raw_review_lengths[i] = unfiltered_length
    neg_sarcastic_review_lengths.append(unfiltered_length)
    mapping[i][0] = 1
    mapping[i][1] = idx

for idx, i in enumerate(pos_y):
    review = reviews[i]["text"].rstrip()
    words = word_tokenize(review)
    unfiltered_length = len(words)
    raw_review_lengths[i] = unfiltered_length
    pos_sarcastic_review_lengths.append(unfiltered_length)
    mapping[i][0] = 2
    mapping[i][1] = idx

for idx, i in enumerate(pos_n):
    review = reviews[i]["text"].rstrip()
    words = word_tokenize(review)
    unfiltered_length = len(words)
    raw_review_lengths[i] = unfiltered_length
    pos_none_review_lengths.append(unfiltered_length)
    mapping[i][0] = 3
    mapping[i][1] = idx

for i,_,_,_ in vader["neutral"]:
    review = reviews[i]["text"].rstrip()
    words = word_tokenize(review)
    unfiltered_length = len(words)
    raw_review_lengths[i] = unfiltered_length

"""
filtered = [word for word in words if word.casefold() not in stop_words]
filtered_length = len(filtered)
filtered_review_lengths.append(filtered_length)
"""

#total_raw = np.array(raw_review_lengths)
#total_filtered = np.array(filtered_review_lengths)
#pos_raw = np.array(pos_raw_review_lengths)
#pos_filtered = np.array(pos_filtered_review_lengths)
#neg_raw = np.array(neg_raw_review_lengths)
#neg_filtered = np.array(neg_filtered_review_lengths)

def get_inliers(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    bound = 1.5 * (q3 - q1)
    return np.logical_and(data >= q1 - bound, data <= q3 + bound)

mask = get_inliers(raw_review_lengths)
idx_to_remove = mapping[np.logical_not(mask), :]

lists = [neg_n, neg_y, pos_y, pos_n]
new_files = ["neg_n2.pkl","neg_y2.pkl","pos_y2.pkl","pos_n2.pkl"]
for i in range(4):
    temp = np.uint32(lists[i])
    temp = temp.delete(idx_to_remove[idx_to_remove[:, 0] == i, 1])
    with open(f"steam_reviews/{new_files[i]}", "wb") as f:
        pickle.dump(temp.tolist(), f)
        

"""
def remove_outliers(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    bound = 1.5 * (q3 - q1)
    return data[np.logical_and(data >= q1 - bound, data <= q3 + bound)]
"""
"""
total_raw = remove_outliers(total_raw)
#total_filtered = remove_outliers(total_filtered)
pos_raw = remove_outliers(pos_raw)
#pos_filtered = remove_outliers(pos_filtered)
neg_raw = remove_outliers(neg_raw)
#neg_filtered = remove_outliers(neg_filtered)

def plot_histogram(data, title):
    plt.hist(data)
    plt.title(title)
    plt.show()

plot_histogram(total_raw, "Unfiltered Word Counts")
#plot_histogram(total_filtered, "Filtered Word Counts")
plot_histogram(pos_raw, "Unfiltered Word Counts (Positive)")
#plot_histogram(pos_filtered, "Filtered Word Counts (Positive)")
plot_histogram(neg_raw, "Unfiltered Word Counts (Negative)")
#plot_histogram(neg_filtered, "Filtered Word Counts (Negative)")
"""