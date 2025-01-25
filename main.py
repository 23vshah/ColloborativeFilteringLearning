import gensim
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

rs = 123

course1 = "this is an introduction data science course which introduces data science to beginners"
course2 = "machine learning for beginners"
courses = [course1, course2]
print(courses)

tokenized_courses = [word_tokenize(course) for course in courses]
print(tokenized_courses)

tokens_dict = gensim.corpora.Dictionary(tokenized_courses)
print(tokens_dict.token2id)

courses_bow = [tokens_dict.doc2bow(course) for course in tokenized_courses]
print(courses_bow)

stop_words = set(stopwords.words('english'))

processed_tokens = [w for w in tokenized_courses[0] if not w.lower() in stop_words]
# print(tokenized_courses[0], processed_tokens)

course_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
course_content_df = pd.read_csv(course_url)
# Merge TITLE and DESCRIPTION title
course_content_df['course_texts'] = course_content_df[['TITLE', 'DESCRIPTION']].agg(' '.join, axis=1)
course_content_df = course_content_df.reset_index()
course_content_df['index'] = course_content_df.index

def tokenize_course(course, keep_only_nouns=True):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(course)
    # Remove English stop words and numbers
    word_tokens = [w for w in word_tokens if (not w.lower() in stop_words) and (not w.isnumeric())]
    # Only keep nouns
    if keep_only_nouns:
        filter_list = ['WDT', 'WP', 'WRB', 'FW', 'IN', 'JJR', 'JJS', 'MD', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS',
                       'RP']
        tags = nltk.pos_tag(word_tokens)
        word_tokens = [word for word, pos in tags if pos not in filter_list]

    return word_tokens

a_course = course_content_df.iloc[0, :]['course_texts']
print(tokenize_course(a_course))

tokenized_courses = [tokenize_course(course_text) for course_text in course_content_df['course_texts']]
tokenized_courses[:1]

# WRITE YOUR CODE HERE
tokens_dict = gensim.corpora.Dictionary(tokenized_courses)

# WRITE YOUR CODE HERE
courses_bow = [tokens_dict.doc2bow(course) for course in tokenized_courses]

doc_index = []
doc_id = []
bags_of_token = []
bow = []

for idx, bag in enumerate(courses_bow):
    for word in bag:
        token = tokens_dict[word[0]]
        doc_index.append(idx)
        doc_id.append(course_content_df['COURSE_ID'][idx])
        bags_of_token.append(token)
        bow.append(word[1])


bow_dicts = {"doc_index": doc_index,
           "doc_id": doc_id,
            "token": bags_of_token,
            "bow": bow}


print(pd.DataFrame(bow_dicts))