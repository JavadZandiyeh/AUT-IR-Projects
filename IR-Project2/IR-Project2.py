#!/usr/bin/env python
# coding: utf-8

# In[94]:


import json
import math
from parsivar import Normalizer, Tokenizer, FindStems
from stopwordsiso import stopwords


# In[95]:


# ---------------------------- preprocessing ----------------------------
# normalizer
my_normalizer = Normalizer()
# tokenizer
my_tokenizer = Tokenizer()
# stemmer
my_stemmer = FindStems()
# stop words
persian_stopwords = stopwords("fa")


# In[96]:


# opening JSON file
# f = open('IR_data_news_small.json')
f = open('IR_data_news_12k.json')
# returns JSON object as a dictionary
documents = json.load(f)
N = len(documents)
# closing file
f.close()


# In[97]:


document_index = {}

list_stem_stopword_token_normal = []

# iterating through the json list
for docID in documents:
    # normalize
    normal = my_normalizer.normalize(documents[docID]["content"])
    # tokenize
    token_normal = my_tokenizer.tokenize_words(normal)
    # remove stopwords
    stopword_token_normal = [t for t in token_normal if t not in persian_stopwords]
    # stemming
    stem_stopword_token_normal = [my_stemmer.convert_to_stem(w) for w in stopword_token_normal]
    list_stem_stopword_token_normal.append(stem_stopword_token_normal)
    
    # --------------------------- document indexing --------------------------
    # creating document index
    docLen = len(stem_stopword_token_normal)
    for pos in range(docLen):
        term = stem_stopword_token_normal[pos]
        if term not in document_index: # first visit of this term in all documents
            document_index[term] = {docID: {'doc_freq': 1}}
        else:
            if docID not in document_index[term]: # first visit of this term in this document 
                document_index[term][docID] = {'doc_freq': 1}
            else: # not first visit of this term in this document
                document_index[term][docID]['doc_freq'] += 1

# compute weight for each term
for term in document_index:
    n_t = len(document_index[term])
    for docID in document_index[term]:
        f_t_d = document_index[term][docID]['doc_freq']
        document_index[term][docID]['weight'] = ((1 + math.log10(f_t_d)) * math.log10(N/n_t))


# In[164]:


# --------------------------- query processing --------------------------
# preprocess query
query = input('کوئری خود را وارد کنید: ')
query_n = my_normalizer.normalize(query)
query_nt = my_tokenizer.tokenize_words(query_n)
ps = stopwords("fa")
query_ntw = [t for t in query_nt if t not in ps]
query_ntws = [my_stemmer.convert_to_stem(w) for w in query_ntw]

# query vector
query_v = {}
for term in query_ntws:
    if term not in query_v:
        query_v[term] = 1
    else:
        query_v[term] += 1

for term in query_v:
    if term not in document_index:
        query_v[term] = 0
    else:
        n_t = len(document_index[term])
        query_v[term] = (1 + math.log10(query_v[term])) * math.log10(N/n_t)

query_vector = {}
for term in query_v:
    if query_v[term] != 0:
        query_vector[term] = query_v[term]

print(query_vector)


# In[165]:


# champion lists
rel_docs = {}
for term in query_vector:
    rel_docs[term] = {}
    for docID in document_index[term]:
        rel_docs[term][docID] = {}
        rel_docs[term][docID]['a_2'] = math.pow(document_index[term][docID]['weight'], 2)
        rel_docs[term][docID]['weight'] = document_index[term][docID]['weight'] * query_vector[term]

rel_docs_1 = {}
for term in rel_docs:
    for docID in rel_docs[term]:
        if docID not in rel_docs_1:
            rel_docs_1[docID] = {}
            rel_docs_1[docID]['sum_a_2'] = rel_docs[term][docID]['a_2']
            rel_docs_1[docID]['weight'] = rel_docs[term][docID]['weight']
        else:
            rel_docs_1[docID]['sum_a_2'] += rel_docs[term][docID]['a_2']
            rel_docs_1[docID]['weight'] += rel_docs[term][docID]['weight']

radical_sum_b_2 = 0
for term in query_vector:
    radical_sum_b_2 += math.pow(query_vector[term], 2)
radical_sum_b_2 = math.sqrt(radical_sum_b_2)

relevant_docs = {}
for docID in rel_docs_1:
    relevant_docs[docID] = (rel_docs_1[docID]['weight'] / (math.sqrt(rel_docs_1[docID]['sum_a_2']) * radical_sum_b_2))

relevant_docs_sorted = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)


# In[166]:


def print_relevant_contents(docID, terms):
    doc = documents[str(docID)]["content"]
    doc_n = my_normalizer.normalize(doc)
    doc_nt = my_tokenizer.tokenize_words(doc_n)
    doc_nts = [my_stemmer.convert_to_stem(w) for w in doc_nt]

    for i in range(len(doc_nts)):
        term = doc_nts[i]
        if term in terms:
            for t in doc_nt[i-7: i+7]:
                print(t, end=" ")
            print(end="\n")

print(relevant_docs_sorted[:5])
k = 0
for doc in relevant_docs_sorted:
    if k >= 5:
        break
    k += 1
    print('title: ')
    print(documents[doc[0]]['title'], end='\n')
    print('content: ')
    print_relevant_contents(doc[0], query_vector.keys())
    print('\n\n')    

