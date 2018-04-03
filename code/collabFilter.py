from surprise import Dataset, evaluate
from surprise import KNNBasic
from surprise import Reader
from surprise import BaselineOnly
from surprise.model_selection import cross_validate
from surprise import accuracy
from collections import defaultdict
import os, io

def get_top3_recommendations(predictions, topN=3):
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
    
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse=True)
        top_recs[uid] = user_ratings[:topN]
    
    return top_recs

def precision_recall_at_k(predictions, k, threshold=9):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

file_path = "../data/10users.csv"
reader = Reader(line_format='user item rating', rating_scale=(0,10), sep=',', skip_lines=1)
print("Loading data from file...")
data = Dataset.load_from_file(file_path, reader=reader)
print("Building training set from data...")
trainingSet = data.build_full_trainset()

sim_options = {
    'name' : 'cosine'
    #'user_based' : False
}
knn = KNNBasic(sim_options=sim_options)
print("Fitting KNN model to training set...")
knn.fit(trainingSet)

print("Building test set from data...")
testSet = trainingSet.build_anti_testset()
print("Making predictions on the test set...")
predictions = knn.test(testSet)
print("Print predictions...")
print("###########################################")
top3_recommendations = get_top3_recommendations(predictions)
print("Num users:", len(top3_recommendations.keys()))
print(top3_recommendations['4826135'])
print("Metrics ...")
RMSE = accuracy.rmse(predictions, verbose=False)
print("RMSE: "+ str(RMSE))
#FCP = accuracy.fcp(predictions, verbose=False) #Not having more than 2 recommendations per user
#print("FCP: "+ str(FCP))
print("Precision and Recall micro-averaging...")
precisions, recalls = precision_recall_at_k(predictions, 1, threshold=5)
precision = sum(precisions.values())
recall = sum(recalls.values())
total = precision+recall
print("Precision: " +str(float(precision)/total))
print("Recall: " +str(float(recall)/total))
