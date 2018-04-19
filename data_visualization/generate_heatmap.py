from collections import defaultdict
import numpy as np
import seaborn as sns
import csv

# Set up seaborn
sns.set()

# Set up numpy arr with user and anime data
f = open('../data/staff.csv', 'r')
reader = csv.reader(f)
users = defaultdict(int)
anime = defaultdict(int)
user_count = 1
anime_count = 1

# Match user IDs and anime IDs to index
for row in reader:
    curr_user = row[0]
    curr_anime = row[1]
    if curr_user != "userID":
        if users[curr_user] == 0:
            users[curr_user] = user_count
            user_count += 1
    if curr_anime != "animeID":
        if anime[curr_anime] == 0:
            anime[curr_anime] = anime_count
            anime_count += 1

total_users = user_count - 1
total_anime = anime_count - 1

data = np.zeros((total_users, total_anime))
f.close()
f = open('../data/staff.csv', 'r')
r = csv.reader(f)
for row in r:
    curr_user = row[0]
    curr_anime = row[1]
    rating = int(row[2])
    if curr_user != "userID" and curr_anime != "animeID":
        userInd = users[curr_user]-1
        animeInd = anime[curr_anime]-1
        data[userInd][animeInd] = rating

ax = sns.heatmap(data, vmin=0, vmax=10)
fig = ax.get_figure()
fig.savefig('staff_heatmap.png')
