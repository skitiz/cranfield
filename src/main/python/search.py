# Kshitij Bantupalli
# Current NCDG Score = 0.63
# 
# How did I approach this?
# 1. Implemented various forms of normalization, stopwords and other small things.
#  Didn't really make a difference to the ndcg score.
# 2. Got rid of the first attempt, started from scratch to tackle tf-idf as it would make the bulk of the difference.
# 3. Finally got it down after weeks of effort and then struggled with cosine similarity. Found the pseudocode in the book 
# which made life much simpler.
# 4. Implemented tf-idf coupled with cosine similarity and reached a score of 0.61.
# 5. Tried two different stemmers. Kept SnowballStemmer cause it had better results.
# 6. Tried using stopwords but it dropped by ndcg score. Maybe I need a custom stopwords library for this.
# 7. Tried removing special chars, dropped my score again. Commented it out.
# 8. Implemented Probabilistic Model Okapi BM-25. Has a score of 0.49.
# 9. Finish
#
# What could I have done if I had more time?
# 1. Probably figure out a way to tackle synonyms. When I tried to figure out which queries had a lower score, they were 
# all jargon to the dataset.
# 2. Try out LM, just to see the difference.

# Changes in ndcg score in chronological order:
# 1. custom stopwords, removing special characters, AND to OR.    | Nothing too major, got it up to   0.198. 
# * Note : First Discussion Post.
# 2. Implemented tf-idf.                                          | Major, got the score up to        0.49.
# 3. Made a separate function to merge tf-idf and cosine.         | Bulk of my score.                 0.61   
# * Note : Discussion Post Update.
# 4. StopWords, Stemming and Special Chars.                       | Increased my score to a           0.63
# 5. BM 25                                                        | Decreased the score to            0.49


# Before I get flagged for plagirism, I'm gonna list every possible source I've used as reference.
# 1. Implementation of the cosine similarity code, is based off the psuedocode in the book. 
#    Kept the same variable names and structure.
# 2. https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# 3. http://www.nltk.org/howto/stem.html
# 4. Special characters isAlNum() taken from a StackOverflow post.


from readers import read_queries, read_documents    # Importing the corpus to read.
import numpy as np                                  # Used this to set the length of scores and length in cosine.
import math                                         # For the tf, idf scores.
from nltk.corpus import stopwords                   # Stopwords.
from nltk.stem.porter import *                      # Porter Stemmer
from nltk.stem.snowball import SnowballStemmer      # Snowball Stemmer.

# Declare the global variables. totalDocs has the # of docs in corpus.
inverted_index = {}
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


# The original TF-IDF function.
def calculate_idf(query):
    uniqueList = {}
    score = np.zeros(totalDocs)
    rankings = []
    idfSc = 0
    for token in query:
        if token not in uniqueList:
            uniqueList[token] = 1
        else:
            uniqueList[token] += 1
    for token in uniqueList:
        documents = inverted_index[token]
        for document in documents:
            docId = document[0]
            freq = document[1]
            idfSc = tf(uniqueList[token]) * idf(freq)
            score[docId] += idfSc
    for i in range(0, len(score)):
        rankings.append([i, score[i]])
    return[i[0] for i in sorted(rankings, key=lambda tup: tup[1], reverse=True)]

# Probabilistic Model
def bm_25(query):
    unique = []
    scores = np.zeros(totalDocs)
    rankings = []
    for token in query:
        if token not in unique:
            unique.append(token)
    for token in unique:
        documents = inverted_index[token]
        for document in documents:
            docId = document[0]
            freq = document[1]
            scores[docId] += idf(freq)
    for i in range(0, len(scores)):
        rankings.append([i, scores[i]])
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
        # return bm_25(indexed_tokens)
        return calculate_tf_idf(indexed_tokens)
        # return calculate_idf(indexed_tokens)

# Stop Words
def stop_words(tokens):
    str = []
    stop_words = set(stopwords.words('english'))
    for token in tokens:
        if token not in stop_words:
            str.append(token)
    return str

# Porter Stemmer.
def porter_stemmer(tokens):
    stemmer = PorterStemmer()
    str = []
    for token in tokens:
        str.append(stemmer.stem(token))
    return str

# Snowball Stemmer.
def snowball_stemmer(tokens):
    str = []
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for token in tokens:
        str.append(stemmer.stem(token))
    return str

# Removes non alphanumeric characters OR special characters basically.
def remove_special_chars(tokens):
    str = []
    for token in tokens:
        if token.isalnum():
           str.append(token)
    return str

# Tokenize the text passed.
def tokenize(text):
    str = []
    str = text.split(" ")          # 0.49 with just TF-IDF | 0.610 with TF-IDF and Cosine | 0.49 with BM.
    str = stop_words(str)          # 0.611 with nltk.corpus
    # str = porter_stemmer(str)    # 0.629 with PorterStemmer
    str = snowball_stemmer(str)    # 0.63 with SnowballStemmer
    # str = remove_special_chars(str) # 0.54 Drops the scores.
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
