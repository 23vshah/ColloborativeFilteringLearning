import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# also set a random state
rs = 123

user_profile_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
user_profile_df = pd.read_csv(user_profile_url)
user_profile_df.head()

feature_names = list(user_profile_df.columns[1:])

scaler = StandardScaler()
user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])
print("mean {} and standard deviation{} ".format(user_profile_df[feature_names].mean(),user_profile_df[feature_names].std()))

features = user_profile_df.loc[:, user_profile_df.columns != 'user']
user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']

distorsions = []
# for k in range(1, 30):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(features)
#     distorsions.append(kmeans.inertia_)

# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(1, 30), distorsions)
# plt.grid(True)
# plt.title('Elbow curve')

kmeans = KMeans(n_clusters=30)
kmeans.fit(features)
cluster_labels = kmeans.labels_
def combine_cluster_labels(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df

print(combine_cluster_labels(user_ids, cluster_labels))

features = user_profile_df.loc[:, user_profile_df.columns != 'user']
user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
feature_names = list(user_profile_df.columns[1:])

print(f"There are {len(feature_names)} features for each user profile.")

