import pickle
import re

# Load steam reviews text and rating
with open("steam_reviews/reviews.pkl", "rb") as f:
    reviews = pickle.load(f)

# Load Vader classification (supposed sarcasm)
with open("steam_reviews/vader.pkl", "rb") as f:
    vader = pickle.load(f)

ignore_nonascii = re.compile(r'[^\x00-\x7f]')
print(re.sub(ignore_nonascii, '', "testj921[f mc/mmv=1-2=ふぁ'"))

def get_stats(idx, reviews, info_str):
    print(f"Stats for {info_str}:")
    vader_pos_sum = 0
    vader_neg_sum = 0
    pos_count = 0
    pos_idx = []
    neg_idx = []
    for i, score, vader_score, num_sent in idx:
        if score > 0:
            vader_pos_sum += vader_score
            pos_count += 1
            pos_idx.append(i)
        else:
            vader_neg_sum += vader_score
            neg_idx.append(i)
    
    print(f"Mean Positive Review Vader Score: {vader_pos_sum / pos_count}")
    print(f"Positive Count: {pos_count}")
    print(f"Mean Negative Review Vader Score: {vader_neg_sum / (len(idx) - pos_count)}")
    print(f"Negative Count: {len(idx) - pos_count}")
    print()
    return pos_idx, neg_idx

pos_idx, neg_idx = get_stats(vader["agree"], reviews, "Agreement Reviews")
with open("steam_reviews/pos_n.pkl", 'wb') as f:
    pickle.dump(pos_idx, f)
with open("steam_reviews/neg_n.pkl", 'wb') as f:
    pickle.dump(neg_idx, f)
pos_idx, neg_idx = get_stats(vader["disagree"], reviews, "Conflict Reviews")
with open("steam_reviews/pos_y.pkl", 'wb') as f:
    pickle.dump(pos_idx, f)
with open("steam_reviews/neg_y.pkl", 'wb') as f:
    pickle.dump(neg_idx, f)
get_stats(vader["neutral"], reviews, "Neutral Reviews")

# try to map sarcasm from one emotion to another
# try to comment on the game dthe review is derived from
# try to look at hours played
