from surprise import Dataset, evaluate
from surprise import KNNBasic
from surprise import Reader
from surprise import BaselineOnly
from surprise.model_selection import cross_validate, KFold
from surprise import accuracy
import surprise
from collections import defaultdict
import os, io



def baseline_recommendations(train):
    mean_dict = {}
    item_dict = train.ir
    for item in item_dict:
        lst = item_dict[item]
        values = []
        for _, score in lst:
            try:
                score = float(score)
                if score != 0:
                   values.append(score)
            except ValueError:
                continue
        mean_dict[item] = values
    for key in mean_dict:
        scores = mean_dict[key]
        if len(scores) == 0:
            scores.append(0)
        mean_dict[key] = sum(scores) / float(len(scores))

    return mean_dict

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


# Load training data
file_path = "../data/train.csv"
reader = Reader(line_format='user item rating', rating_scale=(0,10), sep=',', skip_lines=1)
print("Loading train data from file...")
data = Dataset.load_from_file(file_path, reader=reader)
train_data = data.build_full_trainset()

# Load test data
file_path = "../data/test.csv"
reader = Reader(line_format='user item rating', rating_scale=(0,10), sep=',', skip_lines=1)
print("Loading test data from file...")
data = Dataset.load_from_file(file_path, reader=reader)
test = data.build_full_trainset()
test_data = test.build_testset()

# User-based CF
user_sim_options = {
    'name' : 'cosine',
    'user_based' : True
}
user_knn = KNNBasic(sim_options=user_sim_options)
print("Fitting KNN user model to training set")
user_knn.fit(train_data)

print("Making user-based predictions on the test set...")
user_predictions = user_knn.test(test_data)

print("User-based stats ...")
RMSE = accuracy.rmse(user_predictions, verbose=False)
print("RMSE: "+ str(RMSE))
precisions, recalls = precision_recall_at_k(user_predictions, 15, threshold=7)
precision = sum(precisions.values())
recall = sum(recalls.values())
total = precision+recall
precision = float(precision)/total
recall = float(recall)/total
print("Precision: " +str(precision)
print("Recall: " +str(recall)
print("F1: " + str(float(2*precision*recall)/(precision+recall)))

print("###########################################")
print("###########################################")
print("###########################################")

# Item-based CF
item_sim_options = {
    'name' : 'cosine',
    'user_based' : False
}
item_knn = KNNBasic(sim_options=item_sim_options)
print("Fitting KNN item model to training set")
item_knn.fit(train_data)

print("Making item-based predictions on the test set...")
item_predictions = item_knn.test(test_data)

print("Item-based stats ...")
RMSE = accuracy.rmse(item_predictions, verbose=False)
print("RMSE: "+ str(RMSE))
precisions, recalls = precision_recall_at_k(item_predictions, 15, threshold=7)
precision = sum(precisions.values())
recall = sum(recalls.values())
total = precision+recall
precision = float(precision)/total
recall = float(recall)/total
print("Precision: " +str(precision)
print("Recall: " +str(recall)
print("F1: " + str(float(2*precision*recall)/(precision+recall)))

print("###########################################")
print("###########################################")
print("###########################################")

