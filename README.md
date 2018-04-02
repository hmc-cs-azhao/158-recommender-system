# Anime Recommender System using Collaborative Filtering and MyAnimeList Dataset
Recommender system that applies a user-to-user collaborative filtering algorithm on the MAL dataset to recommend anime for users who have a large enough anime list.

# Installation
We are using the Python Surprise library. You will need to install numpy and surprise:

`pip install numpy` <br />
`pip install scikit-surprise`

# Dataset
The entire set is extracted from members of MAL Updater, the largest club in the MyAnimeList community. The club comprises 29,231 users who have collectively viewed a total of 14,076 series for a total of 12,650,936 ratings. This is approximately 3% completion of the data matrix. Here is the breakdown of ratings per user and ratings per anime visualized.

![alt text](data_visualization/anime_count.png "Histogram of ratings per user")
![alt text](data_visualization/user_count.png "Description goes here")

Reducing to the top 10% of users sorted by ratings per user as well as the top 10% of anime sorted by ratings per anime, we have a total of 2,125,566 ratings, which is about 52% completion.

# Code Files
collabFilter.py :
Performs collaborative filtering on input data. Change the path to the data csv file on line 20 (will parameterize file path later). Outputs the top 3 recommendations for user id 4826135, assuming that this user is part of the input dataset.

generate_pruned_dataset.py :
Requires a pickle file of users, pickle file of anime, and csv file of full dataset. Paths to these can be changed from lines 6-8 respectively. Outputs a csv file with only data points of users in the user pickle file and anime in the anime pickle file.

generate_topN_anime.py :
Optional parameter provided with `-N` flag to give custom percentage, defaults to 10%. Generates pickle file of the top N % of anime sorted by how many ratings an anime has.

generate_topN_users.py :
Optional parameter provided with `-N` flag to give custom percentage, defaults to 10%. Generates pickle file of the top N % of users sorted by how many ratings that user has recorded.

# Data Visualization Files
anime_count.py :
Goes through csv file of data points and plots histogram of count of anime ratings. Path to csv file can be changed on line 8, output figure is saved to the current directory.

user_count.py :
Goes through csv file of data points and plots histogram of count of user ratings. Path to csv file can be changed on line 8, output figure is saved to the current directory.
