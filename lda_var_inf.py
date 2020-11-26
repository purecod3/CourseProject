import math
import numpy as np
import pandas as pd
import random
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
import time
from nltk.corpus import stopwords
from scipy.special import digamma

# def normalize(row):
#     """
#     CURRENTLY NOT USED
#     normalize a row, but not sure what the extra .0000001 term is for
#     which ensures no value is 1
#     """
#     return row / (row.sum() + 0.0000001)

def normalize(input_matrix, axis):
    """
    Normalizes the columns or rows of a 2d input_matrix so they sum to 1
    """

    sums = input_matrix.sum(axis=axis)
    # try:
    #     assert (np.count_nonzero(sums)==np.shape(sums)[0]) # no set should sum to zero
    # except Exception:
    #     raise Exception("Error while normalizing. Sums to zero")
    if axis == 0:
        new_matrix = np.divide(input_matrix, sums[np.newaxis:], out=np.zeros_like(input_matrix),
                               where=sums[np.newaxis:]!=0)
    else:
        new_matrix = np.divide(input_matrix, sums[:, np.newaxis], out=np.zeros_like(input_matrix),
                               where=sums[:, np.newaxis]!=0)
    return new_matrix

def normalize_rows(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """
    row_sums = input_matrix.sum(axis=1) + 0.0000001
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


def normalize_columns(input_matrix):
    """
    Normalizes the columns of a 2d input_matrix so they sum to 1
    """
    col_sums = input_matrix.sum(axis=0) + 0.0000001
    new_matrix = input_matrix / col_sums[np.newaxis :]
    return new_matrix


def evaluate_embeddings(training_data, training_labels, testing_data, testing_labels):
    """
    Evaluation using classification (SVM)
    """
    clf = svm.SVC()
    clf.fit(training_data, training_labels)
    predictions = clf.predict(testing_data)
    # print(predictions)
    # print(testing_labels)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and testing_labels[i] == 1:
            true_positives += 1
        elif predictions[i] == 1 and testing_labels[i] == 0:
            false_positives += 1
        elif predictions[i] == 0 and testing_labels[i] == 1:
            false_negatives += 1
    if true_positives+false_positives > 0:
        print("  Precision = {0:.3f} ({1}/{2})".format(true_positives/(true_positives+false_positives), true_positives, true_positives+false_positives))
    print("  Recall = {0:.3f} ({1}/{2})".format(true_positives/(true_positives+false_negatives), true_positives, true_positives+false_negatives))


def load_csv(input_path, test_size, num_docs, stop_words, min_word_freq, text_column, label_column, label_dict):
    """
    Load csv input file and creates test and training data sets
    """
    # labels = [] # classification labels UNUSED
    # documents = [] # list of documents UNUSED
    df = pd.read_csv(input_path,
                     converters={
                         label_column: lambda x: (label_dict[x]),
                         text_column: lambda line: (re.sub('[^0-9a-zA-Z]+', ' ', line).lower().split()),
                                    # replace non-alphanumeric characters with spaces
                     },
                     encoding='unicode_escape',
                     nrows=num_docs)

    # changed train test split
    training_df , testing_df = train_test_split(df, test_size = 0.2)
    training_df = training_df.reset_index()
    testing_df = testing_df.reset_index()
    
    print('Training data size = {}'.format(training_df.shape[0]))
    print('Testing data size = {}'.format(testing_df.shape[0]))

    # Remove the stop words
    # REVIEW - here stop words is just using the most common words
    # base vocabulary on words found in training set
    word_freq = {}
    for i, row in training_df.iterrows():
        for word in row[text_column]:
            if word in word_freq.keys():
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    words_sorted_by_freq = sorted(word_freq.items(), key=lambda kv: kv[1])
    words_sorted_by_freq = list(filter(lambda kv: kv[1] >= min_word_freq, words_sorted_by_freq))

    # get vocabulary by removing stopwords
    vocabulary = sorted([kv[0] for kv in words_sorted_by_freq if kv[0] not in stop_words])
    words = set(vocabulary) # to be used to check if word in vocabulary
    vocabulary_size = len(vocabulary)

    # get word indices - to be used for creating term doc matrices
    idx = dict(zip(vocabulary, range(vocabulary_size)))
    
    print('Vocabulary size = {}'.format(vocabulary_size))
    # print(idx)

    # create testing term doc matrix
    testing_term_doc_matrix = np.zeros([testing_df.shape[0], vocabulary_size], dtype=np.float)
    for i, row in testing_df.iterrows(): # for each document
        for word in row[text_column]: # for each word in the document
            if word in words: # increment word count if word found in vocabulary words
                testing_term_doc_matrix[i][idx[word]] += 1

    # create training term doc matrix
    training_term_doc_matrix = np.zeros([training_df.shape[0], vocabulary_size], dtype=np.float)
    for i, row in training_df.iterrows(): # for each document
        for word in row[text_column]: # for each word in the document
            if word in words: # increment word count if word found in vocabulary words
                training_term_doc_matrix[i][idx[word]] += 1

    # function return
    return (vocabulary_size, # size of vocabulary
            training_term_doc_matrix, # train features
            training_df[label_column], # train class
            testing_term_doc_matrix, # test features
            testing_df[label_column], # test class
            vocabulary,) # list of words in the vocabulary


class LDA(object):
    def __init__(self, term_doc_matrix, num_docs, vocabulary, vocabulary_size, num_topics,
                 max_e_iter, e_epsilon, max_em_iter, em_epsilon):
        self.term_doc_matrix = term_doc_matrix # |D| x |V|
        self.num_docs = num_docs
        self.vocabulary = vocabulary
        self.vocabulary_size = vocabulary_size
        self.num_topics = num_topics
        self.max_e_iter = max_e_iter 
        self.e_epsilon = e_epsilon
        self.max_em_iter = max_em_iter
        self.em_epsilon = em_epsilon
        self.alpha = None # topic distribution over the whole corpus (): k-sized 1d array
        self.beta = None # word distribution by topic (eq. 47: phi_i_v): k x |V|
        self.pi = None # word distribution by topic for each document
        # self.topic_sampling_count = None

    def initialize_params(self):
        # initialize alpha
        self.alpha = np.array([1/self.num_topics for i in range(self.num_topics)])
        
        # initialize beta - must be equivalent to word assignment to topic
        self.beta = np.ones((self.num_topics, self.vocabulary_size))
        self.beta = normalize(self.beta, 0) # normalize by word as per LDA doc pg. 1005 fig. 6 line 7


    def expectation_step(self):
        # update phi
        # in Chase's paper, this is eq. 45 and it is denoted as pi
        # changed to phi to match the LDA paper

        # # initialize phi: k x |V| - this is just beta
        # phi = np.ones((self.num_topics, self.vocabulary_size))
        # # as per lda [3], normalize by word using number of topics (p. 1005, fig. 6, line 1)
        # phi = normalize(phi, 0)
        
        # initialize gamma: |D| x k
        gamma = np.full((self.num_docs, self.num_topics), self.alpha)
        for i in range(self.num_docs):
            gamma[i] = gamma[i] + self.term_doc_matrix[i].sum()/self.num_topics
        
        for i in range(self.max_e_iter): # TODO: add distance stopping criteria self.e_epsilon or convergence
            p = []
            for j in range(self.num_docs):
                # update pi
                p.append(((self.beta * self.term_doc_matrix[j]).T * np.exp(digamma(gamma[j])-digamma(gamma[j].sum()))).T)

            self.pi = np.array(p) 
            # normalize topic assignment probability by word (p.1005, fig. 6, line 7)
            for j in range(self.num_docs):
                self.pi[j] = normalize(self.pi[j], 0)

            # update gamma
            for j in range(self.num_docs):
                gamma[j] = self.alpha + self.pi[j].sum(axis=1)

    def maximization_step(self):
        # updata beta
        self.beta = self.pi.sum(axis=0)
        self.beta = normalize(self.beta, 0)
        # TODO: confirm beta needs to be normalized
        # not sure if beta needs to be normalized - no mention in either paper    
        # TODO: update alpha
        self.alpha = np.random.random(self.num_topics)
        self.alpha = self.alpha/self.alpha.sum()
    
    ####################################
    # TODO: confirm no longer necessary
    # TODO: Consider making this indices into the dictionary instead of rebuild documents
    # if this is affecting performance somehow.  (probably won't)
    # def get_variable_length_docs(self, term_doc_matrix):
    #     variable_length_docs = []
    #     for row in term_doc_matrix:
    #         variable_length_doc = []
    #         for word, count in enumerate(row.tolist()):
    #             for i in range(int(count)):
    #                 variable_length_doc.append(word)
    #         variable_length_docs.append(variable_length_doc)
    #     return variable_length_docs

    # TODO: confirm no longer needed
    # def train(self, num_topics, term_doc_matrix, iterations, learning_rate=0.1, word_sample_weight=0.5, topic_sample_weight=0.5):
    #     print('Training an LDA model with {} topics...'.format(num_topics))
    #     docs = self.get_variable_length_docs(term_doc_matrix)
    #     num_docs = term_doc_matrix.shape[0]

        # TODO: Fill in.  Might pull in more from lda_with_learning_rate.
        # E-step

        # j - document index
        # t - word index (up to number of words in document j)
        # i - topic index

        # pi_j_t_i: 
        # proportional to: 
        # phi - topic distribution over vocabulary
        # for each document (j) - k x |d_j|

        # exp = e to power of (scipy/ numpy)
        # psi - digamma function (scipy)
        # gamma - distribution of topic over the document
        # get digamma of gamma_j_i and subtract digamma of sum of all j, k topics
        # i needs to add up to 1 (normalize across i)

        # gamma_j_i update:
        # alpha = length k topics - constant - set 1/K topics
        # sum of topic weight of all words in document j pi_j_t_i (what was calculated from previous step)

        # for each doc, get distance using L2 or cosine and compare to epsilon / # of iterations? keep track of distances

        # for each j in docs:

        # M-step
        # recalculate phi - vocabulary word distribution over topic
        # sum up across all docs and words
        # normalize across i (over v) 
        # pass
    ####################################

    def print_model(self):
        print('===Corpus level topic distribution (alpha)===')
        for i in range(self.num_topics):
            print('Topic {}: alpha:{}'.format(i, self.alpha[i]))

        print('===Corpus level word distribution for each topic (beta)===')
        top10_beta = np.argsort(self.beta)[:,-10:]
        print(top10_beta)
        print(len(self.vocabulary))
        print(self.beta.shape)
        for i in range(self.num_topics):
            print('Topic {}:'.format(i))
            for j in range(10):
                print("Word: {}, beta: {}".format(self.vocabulary[top10_beta[i,j]],
                                                  self.beta[i, top10_beta[i,j]]))

    # def print_model(self, vocabulary, print_freq_threshold=0.02):
    #     for topic, words in enumerate(self.phi):
    #         print('Topic {0}: {1:.3f}'.format(topic, self.alpha[topic]))
    #         for w, p in enumerate(words / (np.sum(words) + 0.0000001)):
    #             if p > print_freq_threshold:
    #                 print(' {0} : {1}'.format(vocabulary[w], p))
    
    def lda_em(self):
        self.initialize_params()
        for i in range(self.max_em_iter):
            print('EM algorithm iteration {}...'.format(i))
            self.expectation_step()
            self.maximization_step()
            # TODO: calculate log likelihood?

def main():
    stop_words = set(stopwords.words('english'))
    num_docs = 500
    num_topics = 10
    max_e_iter = 10
    e_epsilon = .001
    max_em_iter = 10
    em_epsilon = .001

    # split training and test, get term doc matrix, set vocabulary and vocabulary size to 
    # ones found in training set
    (vocabulary_size,
    training_term_doc_matrix,
    training_labels,
    testing_term_doc_matrix,
    testing_labels,
    vocabulary) = load_csv(input_path = 'FA-KES-Dataset.csv',
                          test_size=0.2,
                          num_docs = num_docs,
                          stop_words=stop_words,
                          min_word_freq=5,
                          text_column='article_content',
                          label_column='labels',
                          label_dict = {'1': 1, '0': 0})

    # just use the term frequencies as the model features (naive baseline)
    print("== SVM with word frequencies ==")
    evaluate_embeddings(normalize_rows(training_term_doc_matrix),
                        training_labels,
                        normalize_rows(testing_term_doc_matrix),
                        testing_labels)

    # now use the LDA model to create model features (test model)
    print("== SVM with topic distributions from LDA ==")
    lda = LDA(training_term_doc_matrix, len(training_term_doc_matrix), vocabulary, vocabulary_size, num_topics,
              max_e_iter, e_epsilon, max_em_iter, em_epsilon)
    # lda.train(num_topics=20, term_doc_matrix=training_term_doc_matrix, iterations=100, learning_rate=0.1, word_sample_weight=0.6, topic_sample_weight=0.8)
    # TODO: need to figure out where learning rate, word_sample_weight, and topic_sample_weight are used
    lda.lda_em()
    lda.print_model()

    # every document will have a topic distribution
    # use topic distribution instead of word frequencies to classify
    # may need to rewrite this to just return theta (topic distribution by document)
    # evaluate_embeddings(lda.get_topic_distributions(term_doc_matrix=training_term_doc_matrix, iterations=50, learning_rate=0.2),
    #                     training_labels,
    #                     lda.get_topic_distributions(term_doc_matrix=testing_term_doc_matrix, iterations=50, learning_rate=0.2),
    #                     testing_labels)


if __name__ == '__main__':
    main()

# data to use: fake news
# term doc matrix - read csv 
# number of topics: 

# vocabulary = ["apple": 1, "banana": 2, "cherry":3]

# d1= [apple, banana]
# d2 = [cherry, cherry, banana]

# [[0, 1]
# [2, 2, 1]

# phi = [[.3, .4, .3],[.2, .5. .3],[.1, .9, .0]]
