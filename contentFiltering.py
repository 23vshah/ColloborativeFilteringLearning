import pandas as pd
import numpy as np
from sklearn import preprocessing
# also set a random state
rs = 123

course_genres = ['Python', 'Database', 'MachineLearning']
courses = [['Machine Learning with Python', 1, 0, 1], ["SQL with Python", 1, 1, 0]]
courses_df = pd.DataFrame(courses, columns = ['Title'] + course_genres)
print(courses_df)

users = [['user0', 'Machine Learning with Python', 3], ['user1', 'SQL with Python', 2]]
users_df = pd.DataFrame(users, columns = ['User', 'Title', 'Rating'])
u0 = np.array([[3, 0]])
C = courses_df[course_genres].to_numpy()

print(f"User profile vector shape {u0.shape} and course genre matrix shape {C.shape}")

u0_weights = np.matmul(u0, C)

u1 = np.array([[0, 2]])
u1_weights = np.matmul(u1, C)

print(u0_weights, u1_weights)

weights = np.concatenate((u0_weights.reshape(1, 3), u1_weights.reshape(1, 3)), axis=0)
profiles_df = pd.DataFrame(weights, columns=['Python', 'Database', 'MachineLearning'])
profiles_df.insert(0, 'user', ['user0', 'user1'])

new_courses = [['Python 101', 1, 0, 0], ["Database 101", 0, 1, 0], ["Machine Learning with R", 0, 0, 1]]
new_courses_df = pd.DataFrame(new_courses, columns = ['Title', 'Python', 'Database', 'MachineLearning'])

new_courses_df = new_courses_df.loc[:, new_courses_df.columns != 'Title']
course_matrix = new_courses_df.values

profiles_df = profiles_df.loc[:, profiles_df.columns != 'user']
profile_matrix = profiles_df.values

scores = np.matmul(course_matrix, profile_matrix.T)


course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
course_genres_df = pd.read_csv(course_genre_url)
profile_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
profile_df = pd.read_csv(profile_genre_url)

test_users_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/rs_content_test.csv"
test_users_df = pd.read_csv(test_users_url)

test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
test_user_ids = test_users['user'].to_list()
print(f"Total numbers of test users {len(test_user_ids)}")
test_user_profile = profile_df[profile_df['user'] == 1078030]
test_user_vector = test_user_profile.iloc[0, 1:].values


enrolled_courses = test_users_df[test_users_df['user'] == 1078030]['item'].to_list()
enrolled_courses = set(enrolled_courses)
all_courses = set(course_genres_df['COURSE_ID'].values)

unknown_courses = all_courses.difference(enrolled_courses)
unknown_course_genres = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
# Now let's get the course matrix by excluding `COURSE_ID` and `TITLE` columns:
course_matrix = unknown_course_genres.iloc[:, 2:].values

score = np.dot(course_matrix[1], test_user_vector)
print(course_matrix[1], test_user_vector, score)

# We first reload all datasets again, and create an empty dictionary to store the results
test_users_df = pd.read_csv(test_users_url)
profile_df = pd.read_csv(profile_genre_url)
course_genres_df = pd.read_csv(course_genre_url)
res_dict = {}
score_threshold = 10.0


def generate_recommendation_scores():
    users = []
    courses = []
    scores = []
    for user_id in test_user_ids:
        test_user_profile = profile_df[profile_df['user'] == user_id]
        # get user vector for the current user id
        test_user_vector = profile_df[profile_df['user'] == user_id].iloc[0, 1:].values

        # get the unknown course ids for the current user id
        enrolled_courses = test_users_df[test_users_df['user'] == user_id]['item'].to_list()
        unknown_courses = all_courses.difference(enrolled_courses)
        unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
        unknown_course_ids = unknown_course_df['COURSE_ID'].values

        # user np.dot() to get the recommendation scores for each course
        recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

        # Append the results into the users, courses, and scores list
        for i in range(0, len(unknown_course_ids)):
            score = recommendation_scores[i]
            # Only keep the courses with high recommendation score
            if score >= score_threshold:
                users.append(user_id)
                courses.append(unknown_course_ids[i])
                scores.append(recommendation_scores[i])

    return users, courses, scores


profile_df[profile_df['user'] == 1078030].iloc[0, 1:].values
# Return users, courses, and scores lists for the dataframe
users, courses, scores = generate_recommendation_scores()
res_dict['USER'] = users
res_dict['COURSE_ID'] = courses
res_dict['SCORE'] = scores
res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
# Save the dataframe
res_df.to_csv("profile_rs_results.csv", index=False)

print(res_df)
