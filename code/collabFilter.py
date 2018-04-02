from surprise import Dataset, evaluate
from surprise import KNNBasic
from surprise import Reader
from surprise import BaselineOnly
from surprise.model_selection import cross_validate
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

file_path = "../data/half.csv"
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
