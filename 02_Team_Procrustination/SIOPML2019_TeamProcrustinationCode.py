#SIOP ML Competition 2019
#Team Name: Team Procrustination, Bowling Green State University
#Team Members: Feng Guo, Nicholas Howald, Marie Childers, Jordan Dovel, Sami Nesnidal, Andrew Samo & Samuel McAbee

from scipy import stats
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import doc2vec
from gensim.models import Doc2Vec
from sklearn import utils
import re

# Read in raw files
train = pd.read_csv("siop_ml_train_participant.csv")
test = pd.read_csv("siop_ml_test_participant.csv")
test = test.assign(E_Scale_score=0, A_Scale_score=0, O_Scale_score=0, C_Scale_score=0, N_Scale_score=0 )
df = pd.concat([train, test], ignore_index=True, sort=False)

#Combine all open ended question responses into one
df['full_text'] = df['open_ended_1']+' '+df['open_ended_2']+' '+df['open_ended_3']+' '+df['open_ended_4']+' '+df['open_ended_5']


#Pre-processing (not removing stopwords, no lemma, etc.)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  #only keep english words/letters
    words = text.lower().split()  #lowercase
    return ' '.join(words)
df['clean_text'] = df.full_text.apply(clean_text)


##split the data into training and test dataset
train_size=1088
X_train=df.clean_text[0:train_size]
X_test=df.clean_text[train_size:]

###########################  PART I ##############################################
#Train the first bigram bag-of-words model for all five scores, applying ridge regression;
#the alpha (l2 regularization) for ridge regression were different for all five scores

#E trait
y_train=df["E_Scale_score"].iloc[0:train_size]
# Built one pipeline to run the 1+2 gram and tf-idf for the text feature extraction, then run the ridge regression
regress = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer()), ('ridge', Ridge(alpha=.4))])
model_f= regress.fit(X_train, y_train)
e_pred = model_f.predict(X_test)

#A trait
y_train=df["A_Scale_score"].iloc[0:train_size]
regress = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer()), ('ridge', Ridge(alpha=.9))])
model_f= regress.fit(X_train, y_train)
a_pred = model_f.predict(X_test)

#C trait
y_train=df["C_Scale_score"].iloc[0:train_size]
regress = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer()), ('ridge', Ridge(alpha=1))])
model_f= regress.fit(X_train, y_train)
c_pred = model_f.predict(X_test)

#O trait
y_train=df["O_Scale_score"].iloc[0:train_size]
regress = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer()), ('ridge', Ridge(alpha=1.8))])
model_f= regress.fit(X_train, y_train)
o_pred = model_f.predict(X_test)

#N trait
y_train=df["N_Scale_score"].iloc[0:train_size]
regress = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer()), ('ridge', Ridge(alpha=1.2))])
model_f= regress.fit(X_train, y_train)
n_pred = model_f.predict(X_test)

#convert into z scores
e1=stats.zscore(e_pred)
c1=stats.zscore(c_pred)
o1=stats.zscore(o_pred)
n1=stats.zscore(n_pred)
a1=stats.zscore(a_pred)

###########################  PART II  ##############################################
###   Doc2Vec    #####

#First, further clean text using wordnetlemmatizer
from nltk.stem import WordNetLemmatizer
lem= WordNetLemmatizer()
def lemma(text):
    words=nltk.word_tokenize(text)
    return ' '.join([lem.lemmatize(w) for w in words])
df['lem_text'] = df.clean_text.apply(lemma)


#### In order to apply Gensim's Doc2Vec, a label has to be associated with each response
# Here one function was defined to add label (train_i for train dataset, and test_i for test_data) for each combined response
def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled


##split the data into training and test dataset
train_size = 1088
X_train = df['lem_text'][:train_size]
X_test = df['lem_text'][train_size:]


#add the labels
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test  #doc2vec class will be trained on the whole data


# run doc2vec on N trait
y_train = df['N_Scale_score'][:train_size]
vectorsize=2000# 2000 dimensional feature vectors are extracted

# for each trait, the model is trained repeatedly 10 times, then final prediction is based on average of 10 times
n_doc = dict()
for t in range(10):
    #Distributed bag of words (DBOW) approach is applied when dm = 0
    model_dbow = Doc2Vec(dm=0, vector_size=vectorsize, hs=0, negative=5, min_count=2, alpha=0.065, sample=0, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(10):   # train the model for 10 epochs
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    def get_vectors(model, corpus_size, vectors_size, vectors_type):    # extract the trained vectors from doc2vec
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
           prefix = vectors_type + '_' + str(i)
           vectors[i] = model.docvecs[prefix]
        return vectors
    train_vectors_dbow = get_vectors(model_dbow, len(X_train), vectorsize, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), vectorsize, 'Test')

    # run ridge regression
    clf = Ridge(alpha=10000)
    clf.fit(train_vectors_dbow, y_train)
    n_doc[t] = clf.predict(test_vectors_dbow)

# Average over 10 predictions for N trait
n_doc_f=(n_doc[0] + n_doc[1]+n_doc[2]+n_doc[3]+n_doc[4]+n_doc[5]+n_doc[6]+n_doc[7]+n_doc[8]+n_doc[9])/10

#Build similar models for other traits
#E

y_train = df['E_Scale_score'][:train_size]

e_doc = dict()
for t in range(10):
    #Distributed bag of words (DBOW) approach is applied when dm = 0
    model_dbow = Doc2Vec(dm=0, vector_size=vectorsize, hs=0, negative=5, min_count=2, alpha=0.065, sample=0, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(10):   # train the model for 10 epochs
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1) # reshuffle
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), vectorsize, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), vectorsize, 'Test')

    # run ridge regression
    clf = Ridge(alpha=10000)
    clf.fit(train_vectors_dbow, y_train)
    e_doc[t] = clf.predict(test_vectors_dbow)

# Average over 10 predictions for N trait
e_doc_f=(e_doc[0] + e_doc[1]+e_doc[2]+e_doc[3]+e_doc[4]+e_doc[5]+e_doc[6]+e_doc[7]+e_doc[8]+e_doc[9])/10

#C

y_train = df['C_Scale_score'][:train_size]

c_doc = dict()
for t in range(10):
    #Distributed bag of words (DBOW) approach is applied when dm = 0
    model_dbow = Doc2Vec(dm=0, vector_size=vectorsize, hs=0, negative=5, min_count=2, alpha=0.065, sample=0, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(10):   # train the model for 10 epochs
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1) # reshuffle
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), vectorsize, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), vectorsize, 'Test')

    # run ridge regression
    clf = Ridge(alpha=10000)
    clf.fit(train_vectors_dbow, y_train)
    c_doc[t] = clf.predict(test_vectors_dbow)

# Average over 10 predictions for N trait
c_doc_f=(c_doc[0] + c_doc[1]+c_doc[2]+c_doc[3]+c_doc[4]+c_doc[5]+c_doc[6]+c_doc[7]+c_doc[8]+c_doc[9])/10

#O

y_train = df['O_Scale_score'][:train_size]

o_doc = dict()
for t in range(10):
    #Distributed bag of words (DBOW) approach is applied when dm = 0
    model_dbow = Doc2Vec(dm=0, vector_size=vectorsize, hs=0, negative=5, min_count=2, alpha=0.065, sample=0, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(10):   # train the model for 10 epochs
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1) # reshuffle
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), vectorsize, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), vectorsize, 'Test')

    # run ridge regression
    clf = Ridge(alpha=10000)
    clf.fit(train_vectors_dbow, y_train)
    o_doc[t] = clf.predict(test_vectors_dbow)

# Average over 10 predictions for N trait
o_doc_f=(o_doc[0] + o_doc[1]+o_doc[2]+o_doc[3]+o_doc[4]+o_doc[5]+o_doc[6]+o_doc[7]+o_doc[8]+o_doc[9])/10


#A

y_train = df['A_Scale_score'][:train_size]

a_doc = dict()
for t in range(10):
    #Distributed bag of words (DBOW) approach is applied when dm = 0
    model_dbow = Doc2Vec(dm=0, vector_size=vectorsize, hs=0, negative=5, min_count=2, alpha=0.065, sample=0, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(10):   # train the model for 10 epochs
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1) # reshuffle
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), vectorsize, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), vectorsize, 'Test')

    # run ridge regression
    clf = Ridge(alpha=10000)
    clf.fit(train_vectors_dbow, y_train)
    a_doc[t] = clf.predict(test_vectors_dbow)

# Average over 10 predictions for N trait
a_doc_f=(a_doc[0] + a_doc[1]+a_doc[2]+a_doc[3]+a_doc[4]+a_doc[5]+a_doc[6]+a_doc[7]+a_doc[8]+a_doc[9])/10

#transform into z scores
e2=stats.zscore(e_doc_f)
c2=stats.zscore(c_doc_f)
o2=stats.zscore(o_doc_f)
n2=stats.zscore(n_doc_f)
a2=stats.zscore(a_doc_f)


########################### Ensemble model   ##############################################
###  Weighted average between 2gram bag-of-words and doc2vec


e_avg_f=e1*.2+e2*.8
c_avg_f=c1*.8+c2*.2
o_avg_f=o1*.2+o2*.8
n_avg_f=n1*.3+n2*.7
a_avg_f=a1*.2+a2*.8

sub = pd.DataFrame()
sub['E'] = e_avg_f
sub['A'] = a_avg_f
sub['O'] = o_avg_f
sub['C'] = c_avg_f
sub['N'] = n_avg_f
sub.to_csv('final_sub.csv',index=False)