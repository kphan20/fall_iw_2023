import pickle
import random

with open("steam_reviews/pos_y.pkl", "rb") as f:
    pos_y = pickle.load(f)

with open("steam_reviews/reviews.pkl", "rb") as f:
    reviews = pickle.load(f)

unlabelled_results = []
labelled_results = []

rand_ind = list(range(len(pos_y)))
random.shuffle(rand_ind)
for idx, i in enumerate(rand_ind):
    print(f"Review {idx}")
    print(f"Reviews labelled: {len(labelled_results)}")
    print(f"{reviews[i]}")
    result = input()
    if result == 'y':
        labelled_results.append(idx)
    elif result == 'q':
        with open("results.pkl", "wb") as f:
            pickle.dump(labelled_results, f)