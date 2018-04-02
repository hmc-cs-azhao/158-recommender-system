import pickle
import csv

# Paths to top N users and anime
# Path to full recommendation dataset
user_path = "users.pickle"
anime_path = "anime.pickle"
rec_path = "../data/clean-data.csv"

# Unpickle user and anime files
users = pickle.load(open(user_path, 'r'))
anime = pickle.load(open(anime_path, 'r'))

# Write out only data points that have both
# a valid user and anime
output = open("pruned_data.csv", 'w')
writer = csv.writer(output)
f = open(rec_path, 'r')
reader = csv.reader(f)
pruned_cnt = 0
for row in reader:
    user_id = row[0]
    anime_id = row[1]
    if user_id in users and anime_id in anime:
        writer.writerow(row)
        pruned_cnt += 1

output.close()
f.close()

print("Total data points:", pruned_cnt)
print("Percent matrix completion:", pruned_cnt * 1.0/(len(users) * len(anime)))


