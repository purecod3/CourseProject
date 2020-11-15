import numpy as np
import math
import pandas as pd


def normalize(input_matrix):
    """
    Normalizes the columns of a 2d input_matrix so they sum to 1
    """

    col_sums = input_matrix.sum(axis=0)
    new_matrix = input_matrix / col_sums[np.newaxis :]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, pickle_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.pickle_path = pickle_path
        self.term_doc_matrix = None
        self.ratings = [] 
        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.epsilon = None  # word distribution of aspect: |V| * k
        self.s = None        # aspect rating: |D| * k
        self.alpha = None    # aspect weight: |D| * k

    def build_corpus(self):
        """
        Update self.number_of_documents
        """
        df = pd.read_pickle(self.pickle_path)
        print(df)
        for index, row in df.iterrows():
            self.documents.append(row['review_words'])
            self.ratings.append(row['rating'])
            self.number_of_documents += 1


    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        vocabulary = set()
        for document in self.documents:
            for word in document:
                vocabulary.add(word)
        self.vocabulary = list(vocabulary)
        self.vocabulary_size = len(self.vocabulary)


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        idx = {}
        for i, word in enumerate(self.vocabulary):
            idx[word] = i
        # print(idx)
        self.term_doc_matrix = np.zeros([len(self.documents), self.vocabulary_size], dtype=np.float)
        for i, document in enumerate(self.documents):
            for word in document:
                self.term_doc_matrix[i][idx[word]] += 1
        # print(self.term_doc_matrix)


    def initialize(self, number_of_aspects):
        """
        """
        self.epsilon = normalize(np.random.rand(self.vocabulary_size, number_of_aspects))
        self.s = np.zeros([self.number_of_documents, number_of_aspects])
        self.alpha = normalize(np.random.rand(self.number_of_documents, number_of_aspects))
        # print(self.epsilon)
        

    def expectation_step(self, number_of_aspects):
        """ The E-step
        """
        print("E step:")
        # TODO: for each d =, infer alpha and z based on theta using equations 8-11
        # Then compute the aspect ratings s using equation 2

            

    def maximization_step(self, number_of_aspects):
        """ The M-step 
        """
        print("M step:")
        # TODO: update theta using equations 13-19        


    def calculate_likelihood(self, number_of_aspects):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        logp = 0.0
        # TODO: calculate logp using equation 12

        self.likelihoods.append(logp)
        return logp


    def lara(self, number_of_aspects, max_iter, min_logp_change):

        """
        Model aspects.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        self.initialize(number_of_aspects)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            self.expectation_step(number_of_aspects)
            self.maximization_step(number_of_aspects)
            logp = self.calculate_likelihood(number_of_aspects)
            print(logp)
            if abs(current_likelihood - logp) < min_logp_change:
                break
            current_likelihood = logp


def main():
    pkl_path = 'processed_amazon_reviews.pkl'
    corpus = Corpus(pkl_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_aspects = 2
    max_iterations = 50
    min_logp_change = 0.001
    corpus.lara(number_of_aspects, max_iterations, min_logp_change)



if __name__ == '__main__':
    main()
