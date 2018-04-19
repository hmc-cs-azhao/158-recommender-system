from collections import defaultdict
import csv

f = open('../data/staff.csv', 'rb')
reader = csv.reader(f)

users = defaultdict(bool)
anime = defaultdict(bool)

count = 0
for row in reader:
    user_id = row[0]
    if user_id != 'userID':
        anime_id = row[1]
        users[user_id] = True
        anime[anime_id] = True
    count += 1

print(len(users.keys()), " users.")
print(len(anime.keys()), " anime.")
print(count, " ratings.")