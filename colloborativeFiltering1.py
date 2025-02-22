import pandas as pd
rating_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
rating_df = pd.read_csv(rating_url)
rating_sparse_df = rating_df.pivot(index='user', columns='item', values='rating').fillna(0).reset_index().rename_axis(index=None, columns=None)

from surprise import NMF
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

rating_df.to_csv("course_ratings.csv", index=False)
# Read the course rating dataset with columns user item rating
reader = Reader(
        line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))

coruse_dataset = Dataset.load_from_file("course_ratings.csv", reader=reader)

trainset, testset = train_test_split(coruse_dataset, test_size=.3)

print(f"Total {trainset.n_users} users and {trainset.n_items} items in the trainingset")
algo = NMF(verbose=True, random_state=123)

# - Train the NMF on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# - Then compute RMSE
accuracy.rmse(predictions)