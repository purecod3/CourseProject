import math
import numpy as np
import pandas as pd
import random
import re
from scipy.special import digamma, gammaln, psi # gamma function utils
from scipy.special import polygamma
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
    clf = svm.SVC(gamma='auto')
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

# From gensim.  
# https://github.com/RaRe-Technologies/gensim/blob/6c80294ad8df16a878cb6df586c797184b39564a/gensim/models/ldamodel.py#L434
def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype) # keep the same precision as input


class LDA(object):
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.num_topics = None
        self.alpha = None
        self.phi = None

    def get_variable_length_docs(self, term_doc_matrix):
        variable_length_docs = []
        for row in term_doc_matrix:
            variable_length_doc = []
            for word, count in enumerate(row.tolist()):
                for i in range(int(count)):
                    variable_length_doc.append(word)
            variable_length_docs.append(variable_length_doc)
        return variable_length_docs

    # From gensim.  
    # https://github.com/RaRe-Technologies/gensim/blob/6c80294ad8df16a878cb6df586c797184b39564a/gensim/models/ldamodel.py#L434
    def update_alpha(self, gammat, rho):
        """
        Update parameters for the Dirichlet prior on the per-document
        topic weights `alpha` given the last `gammat`.
        Uses Newton's method, described in **Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters.** (http://www.stanford.edu/~jhuang11/research/dirichlet/dirichlet.pdf)
        """
        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N
        dalpha = np.copy(self.alpha)
        gradf = N * (psi(np.sum(self.alpha)) - psi(self.alpha) + logphat)

        c = N * polygamma(1, np.sum(self.alpha))
        q = -N * polygamma(1, self.alpha)

        b = np.sum(gradf / q) / ( 1 / c + np.sum(1 / q))

        dalpha = -(gradf - b) / q

        if all(rho() * dalpha + self.alpha > 0):
            self.alpha += rho() * dalpha
        else:
            print("Warning: updated alpha not positive.")
        print("optimized alpha %s" % list(self.alpha))

        return self.alpha

    def train(self, num_topics, term_doc_matrix, iterations, e_iterations, e_epsilon, alpha_offset=1.0, alpha_chunksize=2000, alpha_decay=0.5):
        print('Training an LDA model with {} topics...'.format(num_topics))
        docs = self.get_variable_length_docs(term_doc_matrix)
        num_docs = term_doc_matrix.shape[0]

        # From gensim.  (adapted)
        # https://github.com/RaRe-Technologies/gensim/blob/6c80294ad8df16a878cb6df586c797184b39564a/gensim/models/ldamodel.py#L434
        num_updates = num_docs
        # rho is the "speed" of updating
        rho = lambda: pow(alpha_offset + num_updates / alpha_chunksize, -alpha_decay)

        self.num_topics = num_topics
        self.alpha = np.repeat(1.0 / num_topics, num_topics)

        # Initialize the hidden variables
        pi = [normalize_rows(np.random.random((len(doc), self.num_topics))) for doc in docs]
            # topic assignment of each word in each doc
            # dim = num_docs * |doc| * num_topics

        gamma = normalize_rows(np.random.random((num_docs, self.num_topics)))
            # topic assignment of each doc
            # dim = num_docs * num_topics

        self.phi = normalize_rows(np.random.random((self.num_topics, self.vocabulary_size)))
            # word distribution of each topic
            # dim = num_topics * vocabulary_size

        for iteration in range(iterations):
            print('E-M iteration {}'.format(iteration))
            # E-step
            for j, doc in enumerate(docs):
                for e_iteration in range(e_iterations):
                    previous_gamma = gamma[j].copy()
                    for t, word in enumerate(doc):
                        x = digamma(gamma[j].sum())
                        for i in range(self.num_topics):
                            pi[j][t][i] = self.phi[i][word] * np.exp(digamma(gamma[j][i]) - x)
                        pi[j] = normalize_rows(pi[j])

                    for i in range(self.num_topics):
                        gamma[j][i] = self.alpha[i] + pi[j][:,i].sum()

                    gamma_diff = np.absolute(gamma[j] - previous_gamma).sum()
                    if gamma_diff < e_epsilon:
                        break
                    # print('iter {0} doc {1} e_iter {2} gamma_diff {3:.7f}'.format(iteration, j, e_iteration, gamma_diff))

            # M-step
            for i in range(self.num_topics):
                for v in range(self.vocabulary_size):
                    self.phi[i][v] = 0.0
                    for j, doc in enumerate(docs):
                        for t, word in enumerate(doc):
                            if word == v:
                                self.phi[i][v] += pi[j][t][i]
            self.phi = normalize_rows(self.phi)
            print(self.phi)
            self.alpha = self.update_alpha(gamma, rho)

    def print_model(self, vocabulary, print_freq_threshold=0.02):
        for topic, words in enumerate(self.phi):
            print('Topic {0}: {1:.3f}'.format(topic, self.alpha[topic]))
            for w, p in enumerate(words / (np.sum(words) + 0.0000001)):
                if p > print_freq_threshold:
                    print(' {0} : {1}'.format(vocabulary[w], p))


    def get_topic_distributions(self, term_doc_matrix, e_iterations, e_epsilon):
        print("Predict topic distributions for the new documents using the existing LDA model...")

        docs = self.get_variable_length_docs(term_doc_matrix)
        num_docs = term_doc_matrix.shape[0]

        # Initialize the hidden variables
        pi = [normalize_rows(np.random.random((len(doc), self.num_topics))) for doc in docs]
            # topic assignment of each word in each doc
            # dim = num_docs * |doc| * num_topics

        gamma = normalize_rows(np.random.random((num_docs, self.num_topics)))
            # topic assignment of each doc
            # dim = num_docs * num_topics

        # E-step
        for j, doc in enumerate(docs):
            for e_iteration in range(e_iterations):
                previous_gamma = gamma[j].copy()
                for t, word in enumerate(doc):
                    x = digamma(gamma[j].sum())
                    for i in range(self.num_topics):
                        pi[j][t][i] = self.phi[i][word] * np.exp(digamma(gamma[j][i]) - x)
                    pi[j] = normalize_rows(pi[j])

                for i in range(self.num_topics):
                    gamma[j][i] = self.alpha[i] + pi[j][:,i].sum()

                gamma_diff = np.absolute(gamma[j] - previous_gamma).sum()
                if gamma_diff < e_epsilon:
                    break

        return normalize_rows(gamma)


def main():
    (vocabulary_size,
     training_term_doc_matrix,
     training_labels,
     testing_term_doc_matrix,
     testing_labels,
     vocabulary) = load_csv(input_path = 'FA-KES-Dataset.csv',
                            test_set_size=100,
                            training_set_size=200,
                            num_stop_words=50,
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
    lda.train(num_topics=10, term_doc_matrix=training_term_doc_matrix, iterations=20, e_iterations=10, e_epsilon=0.1)
    lda.print_model(vocabulary, print_freq_threshold=0.01)

    evaluate_embeddings(lda.get_topic_distributions(term_doc_matrix=training_term_doc_matrix, e_iterations=50, e_epsilon=0.01),
                        training_labels,
                        lda.get_topic_distributions(term_doc_matrix=testing_term_doc_matrix, e_iterations=50, e_epsilon=0.01),
                        testing_labels)


if __name__ == '__main__':
    main()
