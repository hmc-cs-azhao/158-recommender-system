from collections import defaultdict
import pickle
import argparse
import csv

file_path = "../data/userSubset907.csv"
f = open(file_path, 'rb')
reader = csv.reader(f)

parser = argparse.ArgumentParser(description='Get the top N users based on number of series seen.')
parser.add_argument("--N", default=1400)
args = parser.parse_args()
N = args.N

user_anime_count = defaultdict(int)
isHeader = True
for row in reader:
    if isHeader:
        isHeader = False
        continue

    user_id = row[0]
    user_anime_count[user_id] += 1

keys = user_anime_count.keys()
pairs = [(key, user_anime_count[key]) for key in keys]
pairs = sorted(pairs, key=lambda pair: pair[1])

users = [user for (user,anime) in pairs[:N]]
output = open('users.pickle', 'w')
pickle.dump(users,output)
output.close()
