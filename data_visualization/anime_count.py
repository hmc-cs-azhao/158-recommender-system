from collections import defaultdict
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

import numpy as np
import csv

file_path = "../data/staff.csv"
f = open(file_path, 'rb')
reader = csv.reader(f)

user_anime_count = defaultdict(int)
isHeader = True
for row in reader:
    if isHeader:
        isHeader = False
        continue
    user_id = row[0]
    user_anime_count[user_id] += 1

anime_counts = user_anime_count.values()
sorted_counts = sorted(anime_counts)
#index = 0
#for i in range(len(sorted_counts)):
#    if sorted_counts[i] >= 2000:
#        index = i
#        break

plt.hist(sorted_counts)#, 40) #[:index],40)
#for i in range(len(n)):
#    print(int(bins[i]), ":", n[i])

plt.xlabel('Anime/User')
plt.ylabel('Count')
plt.savefig('anime_count.png')
