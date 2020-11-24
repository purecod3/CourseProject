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

        self.num_topics = num_topics
        self.alpha = np.repeat(1.0 / num_topics, num_topics)
        self.beta = 1.0 / self.vocabulary_size

        # Initialize the arrays to 0
        z = [[0 for w in range(len(doc))] for doc in docs]
            # topic for each word in each doc
            # dim = num_docs * doc length
        theta = np.zeros((num_docs, self.num_topics))
            # topic distribution of each doc
            # dim = num_docs * num_topics
        phi = np.zeros((self.num_topics, self.vocabulary_size))
            # word distribution of each topic
            # dim = num_topics * vocabulary_size
        self.phi = phi

        doc_sampling_count = np.zeros((num_docs))
        self.topic_sampling_count = np.zeros((self.num_topics))

        # Randomly assign a topic to each word in each doc
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                topic = random.randint(0, self.num_topics-1)
                z[d][n] = topic
                theta[d][topic] += 1
                phi[topic][w] += 1
                doc_sampling_count[d] += 1
                self.topic_sampling_count[topic] += 1

        for iteration in range(iterations):
            topic_changes = 0
            for d, doc in enumerate(docs):
                for n, w in enumerate(doc):
                    topic = z[d][n]

                    # Remove the topic assignment for the word
                    theta[d][topic] -= 1
                    phi[topic][w] -= 1
                    self.topic_sampling_count[topic] -= 1

                    # Recalculate the topic
                    p_topic_given_doc = normalize(theta[d]) * topic_sample_weight + self.alpha * (1-topic_sample_weight)
                    p_word_given_topic = (self.phi[:,w] * (1-word_sample_weight) + phi[:,w] * word_sample_weight + self.beta) / (self.topic_sampling_count + self.vocabulary_size * self.beta) # mixing phi to speed up learning
                    p_topic = normalize(p_topic_given_doc * p_word_given_topic)
                    new_topic = np.random.multinomial(1, p_topic).argmax()

                    # Assign the new topic to the word
                    z[d][n] = new_topic
                    theta[d][new_topic] += 1
                    phi[new_topic][w] += 1
                    self.topic_sampling_count[new_topic] += 1

                    if topic != new_topic:
                        topic_changes += 1

            self.phi = self.phi * (1.0-learning_rate) + phi * learning_rate
            self.alpha = self.alpha * (1.0-learning_rate) + np.average(normalize_rows(theta), axis=0) * learning_rate
            if iteration % 10 == 0:
                print('Iteration {0}: {1} words changed topics.'.format(iteration, topic_changes))


    def print_model(self, vocabulary, print_freq_threshold=0.02):
        for topic, words in enumerate(self.phi):
            print('Topic {0}: {1:.3f}'.format(topic, self.alpha[topic]))
            for w, p in enumerate(words / (np.sum(words) + 0.0000001)):
                if p > print_freq_threshold:
                    print(' {0} : {1}'.format(vocabulary[w], p))


    def get_topic_distributions(self, term_doc_matrix, iterations=10, learning_rate=0.2):
        print("Predict topic distributions for the new documents using the existing LDA model...")
        docs = self.get_variable_length_docs(term_doc_matrix)
        num_docs = term_doc_matrix.shape[0]

        # Initialize the arrays to 0
        z = [[0 for w in range(len(doc))] for doc in docs]
        theta = np.zeros((num_docs, self.num_topics))

        doc_sampling_count = np.zeros((num_docs))

        # Randomly assign a topic to each word in each doc
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                topic = random.randint(0, self.num_topics-1)
                z[d][n] = topic
                theta[d][topic] += 1
                doc_sampling_count[d] += 1

        posterior_topic_distributions = np.tile(self.alpha, (num_docs, 1))
        for iteration in range(iterations):
            for d, doc in enumerate(docs):
                for n, w in enumerate(doc):
                    topic = z[d][n]

                    # Remove the topic assignment for the word
                    theta[d][topic] -= 1

                    # Recalculate the topic
                    p_topic_given_doc = normalize(theta[d]) + self.alpha
                    p_word_given_topic = (self.phi[:,w] + self.beta) / (self.topic_sampling_count + self.vocabulary_size * self.beta + 0.0000001)
                    p_topic = normalize(p_topic_given_doc * p_word_given_topic)
                    new_topic = np.random.multinomial(1, p_topic).argmax()

                    # Assign the new topic to the word
                    z[d][n] = new_topic
                    theta[d][new_topic] += 1
            posterior_topic_distributions = posterior_topic_distributions * (1.0-learning_rate) + normalize_rows(theta) * learning_rate

        # print(posterior_topic_distributions)
        return normalize_rows(posterior_topic_distributions)

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
