import csv
import random

# Open files
f = open('../data/pruned_data5.csv', 'rb')
train_path = open('../data/train.csv', 'wb')
test_path = open('../data/test.csv', 'wb')

# Create CSV writers and reader
f_reader = csv.reader(f)
train_writer = csv.writer(train_path)
test_writer = csv.writer(test_path)

for row in f_reader:
    if random.random() < 0.8:
        train_writer.writerow(row)
    else:
        test_writer.writerow(row)

f.close()
train_path.close()
test_path.close()