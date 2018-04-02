from collections import defaultdict
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import csv

file_path = "../../data/userSubset907.csv"
f = open(file_path, 'rb')
reader = csv.reader(f)

anime_user_count = defaultdict(int)
isHeader = True
for row in reader:
    if isHeader:
        isHeader = False
        continue
    anime_id = row[1]
    anime_user_count[anime_id] += 1

user_counts = anime_user_count.values()
sorted_counts = sorted(user_counts)
index = 0
for i in range(len(sorted_counts)):
    if sorted_counts[i] >= 5000:
        index = i
        break

n, bins, patches = plt.hist(sorted_counts[:index],40)
for i in range(len(n)):
    print(int(bins[i]), ":", n[i])

plt.plot(bins)
plt.xlabel('User/Anime')
plt.ylabel('Count')
plt.savefig('user_count.png')
