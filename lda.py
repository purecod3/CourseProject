import math
import numpy as np
import pandas as pd
import random
import re
from sklearn import svm
import time


def normalize_rows(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """
    row_sums = input_matrix.sum(axis=1) + 0.000001
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


def normalize_columns(input_matrix):
    """
    Normalizes the columns of a 2d input_matrix so they sum to 1
    """
    col_sums = input_matrix.sum(axis=0) + 0.000001
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
    print("  Precision = {0:.3f} ({1}/{2})".format(true_positives/(true_positives+false_positives), true_positives, true_positives+false_positives))
    print("  Recall = {0:.3f} ({1}/{2})".format(true_positives/(true_positives+false_negatives), true_positives, true_positives+false_negatives))


def load_csv(input_path, label_dict, test_set_size):
    labels = []
    documents = []
    df = pd.read_csv(input_path,
                     converters={
                         'Category': lambda x: (label_dict[x]),
                         'Message': lambda x: (re.sub('[^0-9a-zA-Z]+', ' ', x).split()), # replace non-alphanumeric characters with spaces
                     })
    testing_df = df[:test_set_size]
    training_df = df[test_set_size:]

    print('Training data size = {}'.format(training_df.shape[0]))
    print('Testing data size = {}'.format(testing_df.shape[0]))

    words = set()
    for i, row in training_df.iterrows():
        words.update(row['Message'])
    vocabulary = sorted(words)
    vocabulary_size = len(vocabulary)
    idx = dict(zip(vocabulary, range(len(vocabulary))))
    print('Vocabulary size = {}'.format(len(words)))
    # print(idx)

    testing_term_doc_matrix = np.zeros([testing_df.shape[0], vocabulary_size], dtype=np.float)
    for i, row in testing_df.iterrows():
        for word in row['Message']:
            if word in words:
                # the words that do not occur in the training data are skipped
                testing_term_doc_matrix[i][idx[word]] += 1

    training_term_doc_matrix = np.zeros([training_df.shape[0], vocabulary_size], dtype=np.float)
    for i, row in training_df.iterrows():
        for word in row['Message']:
            training_term_doc_matrix[i-test_set_size][idx[word]] += 1

    return (vocabulary_size,
            training_term_doc_matrix,
            training_df['Category'],
            testing_term_doc_matrix,
            testing_df['Category'])


def calculate_topic_distributions(vocabulary_size, num_topics, training_term_doc_matrix, testing_term_doc_matrix):
    # Step 1: Run E-M on training_term_doc_matrix
    # Step 2: Run the E-step using the phi calculated in step 1 to compute gamma for testing_term_doc_matrix
    # return (training_data_topic_distributions, testing_topic_distributions)
    return (training_term_doc_matrix, testing_term_doc_matrix)


def main():
    (vocabulary_size, training_term_doc_matrix, training_labels, testing_term_doc_matrix, testing_labels) = load_csv(input_path = 'spam.csv.1000', label_dict = {'spam': 1, 'ham': 0}, test_set_size = 500)

    print("SVM with word frequencies")
    evaluate_embeddings(normalize_rows(training_term_doc_matrix),
                        training_labels,
                        normalize_rows(testing_term_doc_matrix),
                        testing_labels)

    (training_topic_distributions, testing_topic_distributions) = calculate_topic_distributions(vocabulary_size, 10, training_term_doc_matrix, testing_term_doc_matrix)
    print("SVM with topic distributions from LDA")
    evaluate_embeddings(training_topic_distributions,
                        training_labels,
                        testing_topic_distributions,
                        testing_labels)


if __name__ == '__main__':
    main()
