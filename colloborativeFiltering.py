import numpy as np
import math
import pandas as pd


# An example similarity array stores the similarity of user2, user3, user4, and user5 to user6

knn_sims = np.array([0.8, 0.92, 0.75, 0.83])
knn_ratings = np.array([3.0, 3.0, 2.0, 3.0])
r_u6_ml =  np.dot(knn_sims, knn_ratings)/ sum(knn_sims)
true_rating = 3.0
rmse = math.sqrt(true_rating - r_u6_ml) ** 2

rating_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
rating_df = pd.read_csv(rating_url)
# rating_df.head()

rating_sparse_df = rating_df.pivot(index='user', columns='item', values='rating').fillna(0).reset_index().rename_axis(index=None, columns=None)


from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k', prompt=False)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous KNNBasic algorithm.
algo = KNNBasic()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)