# Kshitij Bantupalli
# Text Mining

from readers import read_queries, read_documents
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer


inverted_index = {}
doc_length = {}
totalDocs = len(read_documents()) + 1

#Removes non indexed tokens.
def remove_not_indexed_toknes(tokens):
    return [token for token in tokens if token in inverted_index]

# Return the cosine scores for each token.
def ranking(scores, Qlength, length):
    ranking = []
    cos_score = 0
    # Scale the cosine score.
    for i in range(0, len(scores)):
        if scores[i] > 0:
            cos_score = scores[i] / ((Qlength ** 0.5) * (length[i] ** 0.5))
        else:
            cos_score = scores[i]
        ranking.append((i, cos_score))
    return ranking

# Returns the tf score of the token.
def tf(freq):
    return 1 + math.log(float(freq))

# Returns the df score of the token.
def idf(freq):
    return math.log((totalDocs-1) / float(freq))

# Returns the docId's to the eval function.
def calculate_tf_idf(query):
    dictionary = {}
    scores = np.zeros(totalDocs)
    length =  np.zeros(totalDocs)
    Qlength = 0
    rankings = []
    for token in query:
        if token in dictionary:
            dictionary[token] += 1
        else:
            dictionary[token] = 1
    # Implemented based on the pseudocode on cosine similarity from the Text Mining book.
    for token in dictionary:
        documents = inverted_index[token]
        noOfDocuments = len(documents)
        vectorQuery = tf(dictionary[token]) * idf(noOfDocuments)
        Qlength = vectorQuery**2
        for document in documents:
            docId = document[0]
            freq = document[1]
            vectorDocument = tf(freq)
            length[docId] += vectorDocument**2
            scores[docId] += vectorDocument * vectorQuery
    rankings = ranking(scores, Qlength, length)
    return [i[0] for i in sorted(rankings, key=lambda tup: tup[1], reverse=True)]

# Calculate tf_idf scores for query.
def search_query(query):
    tokens = tokenize(str(query['query']))
    indexed_tokens = remove_not_indexed_toknes(tokens)
    if len(indexed_tokens) == 0:
        return []
    elif len(indexed_tokens) == 1:
        return inverted_index[indexed_tokens[0]]
    else:
        # return rank_postings(indexed_tokens)
        return calculate_tf_idf(indexed_tokens)

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/ | Stop Words
def stop_words(tokens):
    str = []
    stop_words = set(stopwords.words('english'))
    for token in tokens:
        if token not in stop_words:
            str.append(token)
    return str

# http://www.nltk.org/howto/stem.html
def porter_stemmer(tokens):
    stemmer = PorterStemmer()
    str = []
    for token in tokens:
        str.append(stemmer.stem(token))
    return str

# http://www.nltk.org/howto/stem.html
def snowball_stemmer(tokens):
    str = []
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for token in tokens:
        str.append(stemmer.stem(token))
    return str

# Tokenize the text passed.
def tokenize(text):
    str = []
    str = text.split(" ")          # 0.610 with TF-IDF and Cosine.
    str = stop_words(str)          # 0.611 with nltk.corpus
    # str = porter_stemmer(str)    # 0.629 with PorterStemmer
    str = snowball_stemmer(str)    # 0.63 with SnowballStemmer
    return str

# Add the token to the inverted index.
def add_token_to_index(token, doc_id):
    if token in inverted_index:
        current_postings = inverted_index[token]
        insert = False
        for i in range(0, len(current_postings)):
            if doc_id == current_postings[i][0]:
                current_postings[i][1] += 1
                insert = True
        if (insert == False):
            current_postings.append([doc_id, 1])
            current_postings.sort(key=lambda tup: tup[1])
    else:
        inverted_index[token] = [[doc_id, 1]]

# Create the inverted index.
def add_to_index(document):
    # Extending the search to the body.
    docId = document['id']
    tokens = tokenize(document['title'])
    body = tokenize(document['body'])
    tokens.extend(body)
    for token in tokens:
        add_token_to_index(token, docId)


def create_index():
    for document in read_documents():
        add_to_index(document)


create_index()

if __name__ == '__main__':
    all_queries = [query for query in read_queries() if query['query number'] != 0]
    for query in all_queries:
        documents = search_query(query)
        # print ("Query:{} and Results:{}", format(query, documents))
