import math
import numpy as np
import pandas as pd
import random
import re
from sklearn import svm
import time

def normalize(row):
    return row / (row.sum() + 0.0000001)


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


def load_csv(input_path, test_set_size, training_set_size, num_stop_words, min_word_freq, text_column, label_column, label_dict):
    labels = []
    documents = []
    df = pd.read_csv(input_path,
                     converters={
                         label_column: lambda x: (label_dict[x]),
                         text_column: lambda line: (re.sub('[^0-9a-zA-Z]+', ' ', line).lower().split()),
                                    # replace non-alphanumeric characters with spaces
                     },
                     encoding='unicode_escape',
                     nrows=test_set_size+training_set_size)
    testing_df = df[:test_set_size]
    training_df = df[test_set_size:]

    print('Training data size = {}'.format(training_df.shape[0]))
    print('Testing data size = {}'.format(testing_df.shape[0]))

    # Remove the stop words
    word_freq = {}
    for i, row in training_df.iterrows():
        for word in row[text_column]:
            if word in word_freq.keys():
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    words_sorted_by_freq = sorted(word_freq.items(), key=lambda kv: kv[1])[:-num_stop_words]
    words_sorted_by_freq = list(filter(lambda kv: kv[1] >= min_word_freq, words_sorted_by_freq))

    vocabulary = sorted([kv[0] for kv in words_sorted_by_freq])
    words = set(vocabulary)
    vocabulary_size = len(vocabulary)
    idx = dict(zip(vocabulary, range(len(vocabulary))))
    print('Vocabulary size = {}'.format(len(words)))
    # print(idx)

    testing_term_doc_matrix = np.zeros([testing_df.shape[0], vocabulary_size], dtype=np.float)
    for i, row in testing_df.iterrows():
        for word in row[text_column]:
            if word in words:
                testing_term_doc_matrix[i][idx[word]] += 1

    training_term_doc_matrix = np.zeros([training_df.shape[0], vocabulary_size], dtype=np.float)
    for i, row in training_df.iterrows():
        for word in row[text_column]:
            if word in words:
                training_term_doc_matrix[i-test_set_size][idx[word]] += 1

    return (vocabulary_size,
            training_term_doc_matrix,
            training_df[label_column],
            testing_term_doc_matrix,
            testing_df[label_column],
            vocabulary,)


class LDA(object):
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.num_topics = None
        self.alpha = None
        self.beta = None
        self.phi = None
        self.topic_sampling_count = None

    # TODO: Consider making this indices into the dictionary instead of rebuild documents
    # if this is affecting performance somehow.  (probably won't)
    def get_variable_length_docs(self, term_doc_matrix):
        variable_length_docs = []
        for row in term_doc_matrix:
            variable_length_doc = []
            for word, count in enumerate(row.tolist()):
                for i in range(int(count)):
                    variable_length_doc.append(word)
            variable_length_docs.append(variable_length_doc)
        return variable_length_docs

    def train(self, num_topics, term_doc_matrix, iterations, learning_rate=0.1, word_sample_weight=0.5, topic_sample_weight=0.5):
        print('Training an LDA model with {} topics...'.format(num_topics))
        docs = self.get_variable_length_docs(term_doc_matrix)
        num_docs = term_doc_matrix.shape[0]

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
        pass

    def print_model(self, vocabulary, print_freq_threshold=0.02):
        for topic, words in enumerate(self.phi):
            print('Topic {0}: {1:.3f}'.format(topic, self.alpha[topic]))
            for w, p in enumerate(words / (np.sum(words) + 0.0000001)):
                if p > print_freq_threshold:
                    print(' {0} : {1}'.format(vocabulary[w], p))

def main():
    (vocabulary_size,
     training_term_doc_matrix,
     training_labels,
     testing_term_doc_matrix,
     testing_labels,
     vocabulary) = load_csv(input_path = 'FA-KES-Dataset.csv',
                            test_set_size=100,
                            training_set_size=100,
                            num_stop_words=30,
                            min_word_freq=5,
                            text_column='article_content',
                            label_column='labels',
                            label_dict = {'1': 1, '0': 0})

    print("== SVM with word frequencies ==")
    evaluate_embeddings(normalize_rows(training_term_doc_matrix),
                        training_labels,
                        normalize_rows(testing_term_doc_matrix),
                        testing_labels)

    print("== SVM with topic distributions from LDA ==")
    lda = LDA(vocabulary_size)
    lda.train(num_topics=20, term_doc_matrix=training_term_doc_matrix, iterations=100, learning_rate=0.1, word_sample_weight=0.6, topic_sample_weight=0.8)
    lda.print_model(vocabulary)

    evaluate_embeddings(lda.get_topic_distributions(term_doc_matrix=training_term_doc_matrix, iterations=50, learning_rate=0.2),
                        training_labels,
                        lda.get_topic_distributions(term_doc_matrix=testing_term_doc_matrix, iterations=50, learning_rate=0.2),
                        testing_labels)


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
