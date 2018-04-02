from collections import defaultdict
import pickle
import argparse
import csv

file_path = "../data/userSubset907.csv"
f = open(file_path, 'rb')
reader = csv.reader(f)

parser = argparse.ArgumentParser(description='Get the top N anime based on how many users have seen it.')
parser.add_argument("--N", default=1000)
args = parser.parse_args()
N = args.N

anime_user_count = defaultdict(int)
isHeader = True
for row in reader:
    if isHeader:
        isHeader = False
        continue

    anime_id = row[1]
    anime_user_count[anime_id] += 1

keys = anime_user_count.keys()
pairs = [(key, anime_user_count[key]) for key in keys]
pairs = sorted(pairs, key=lambda pair: pair[1])

anime = [series for (series,cnt) in pairs[:N]]
output = open('anime.pickle', 'w')
pickle.dump(anime,output)
output.close()
