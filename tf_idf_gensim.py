import gensim
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import os

file_docs = []

with open('document1.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:", len(file_docs))
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]
dictionary = gensim.corpora.Dictionary(gen_docs)
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)

# Corrected part: building the index
if not os.path.exists('workdir/'):
    os.makedirs('workdir/')
sims = gensim.similarities.Similarity('workdir/', tf_idf[corpus], num_features=len(dictionary))

file2_docs = []

with open('document2.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:", len(file2_docs))

query_doc = [[w.lower() for w in word_tokenize(text)] for text in file2_docs]

# dictionary = gensim.corpora.Dictionary(query_doc)

query_doc_bow = [dictionary.doc2bow(doc) for doc in query_doc]

print(query_doc_bow)

# perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]
# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])

sum_of_sims = (np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)

# TODO: Keep the results in percentage under 100%
percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')
