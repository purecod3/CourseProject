# data to use: fake news
# number of topics: 

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


vocabulary = ["apple": 1, "banana": 2, "cherry":3]

d1= [apple, banana]
d2 = [cherry, cherry, banana]

[[0, 1]
[2, 2, 1]

phi = [[.3, .4, .3],[.2, .5. .3],[.1, .9, .0]]


for each j in docs:

# M-step
# recalculate phi - vocabulary word distribution over topic
# sum up across all docs and words
# normalize across i (over v) 