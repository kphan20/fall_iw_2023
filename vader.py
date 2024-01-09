from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from util import read_steam_reviews
from nltk.tokenize import sent_tokenize
import pickle

def main(use_cached=True):
    sid_obj = SentimentIntensityAnalyzer()
    review_dicts = read_steam_reviews()

    agreement_ind = []
    conflict_ind = []
    neutral_ind = []

    if use_cached:
        with open("steam_reviews/vader.pkl", "rb") as f:
            vader = pickle.load(f)
        agreement_ind = vader["agree"]
        conflict_ind = vader["disagree"]
        neutral_ind = vader["neutral"]
    else:
        for review_ind, review_dict in enumerate(review_dicts):
            review_sentences = sent_tokenize(review_dict["text"])
            sentiment_sum = 0
            
            if len(review_sentences) > 0:
                for sentence in review_sentences:
                    scores = sid_obj.polarity_scores(sentence)
                    sentiment_sum += scores["compound"]
                
                # if average compound score and review score are the same sign and
                # non-neutral, then the product should be greater than 0.05
                mean_sentiment = sentiment_sum / len(review_sentences)
                sentiment_product = mean_sentiment * review_dict["score"]
                
                # review index, steam score, mean Vader sentiment, num sentences
                review_summary = (review_ind, review_dict["score"], mean_sentiment, len(review_sentences))
                if sentiment_product > 0.05:
                    agreement_ind.append(review_summary)
                elif sentiment_product < 0.05:
                    conflict_ind.append(review_summary)
                else:
                    neutral_ind.append(review_summary)
        
        with open("steam_reviews/vader.pkl", "wb") as f:
            pickle.dump({"agree": agreement_ind, "disagree": conflict_ind, "neutral": neutral_ind}, f)
    
    print("Agree Index Stats:")
    pos_count = 0
    neg_count = 0
    overall_sent_len = 0
    pos_sent_len = 0
    neg_sent_len = 0
    for agreement in agreement_ind:
        if agreement[1] == 1:
            pos_count += 1
            pos_sent_len += agreement[3]
        else:
            neg_count += 1
            neg_sent_len += agreement[3]
        
        overall_sent_len += agreement[3]
    
    print(f"Positive Review Count: {pos_count}")
    print(f"Negative Review Count: {neg_count}")
    print(f"Mean Positive Sentence Length: {pos_sent_len / pos_count}")
    print(f"Mean Negative Sentence Length: {neg_sent_len / neg_count}")
    
    print("Conflict Index Stats:")
    pos_count = 0
    neg_count = 0
    overall_sent_len = 0
    pos_sent_len = 0
    neg_sent_len = 0
    for conflict in conflict_ind:
        if conflict[1] == 1:
            pos_count += 1
            pos_sent_len += conflict[3]
        else:
            neg_count += 1
            neg_sent_len += conflict[3]
        
        overall_sent_len += conflict[3]
    
    print(f"Positive Review Count: {pos_count}")
    print(f"Negative Review Count: {neg_count}")
    print(f"Mean Positive Sentence Length: {pos_sent_len / pos_count}")
    print(f"Mean Negative Sentence Length: {neg_sent_len / neg_count}")
    
    print(f"Mean Overall Sentence Length: {overall_sent_len / (len(agreement_ind) + len(conflict_ind))}")
        
        

if __name__ == '__main__':
    main()



