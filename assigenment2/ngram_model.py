#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    ### YOUR CODE HERE

    for sentence in dataset:
        for idx in range(len(sentence)):

            token_count += 1

            #unigram
            unigram_counts[sentence[idx]] = unigram_counts.get(sentence[idx],0) + 1

            #bigram
            if idx > 0:
                bigram_counts[(sentence[idx-1],sentence[idx])] \
                    = bigram_counts.get((sentence[idx-1],sentence[idx]),0) +1
            else:
                bigram_counts[(-1, sentence[idx])] \
                    = bigram_counts.get((-1, sentence[idx]), 0) + 1

            #trigram
            if idx > 1:
                trigram_counts[(sentence[idx-2],sentence[idx-1],sentence[idx])] \
                    = trigram_counts.get((sentence[idx-2],sentence[idx-1],sentence[idx]),0) +1
            elif idx > 0:
                trigram_counts[(-1,sentence[idx-1],sentence[idx])] \
                    = trigram_counts.get((-1,sentence[idx-1],sentence[idx]),0) +1
            else:
                trigram_counts[(-1,-1,sentence[idx])] \
                    = trigram_counts.get((-1,-1,sentence[idx]),0) +1

    #add sentence starts
    sentence_starts = dataset.shape[0]
    unigram_counts[-1]= sentence_starts
    bigram_counts[(-1,-1)] = sentence_starts
    trigram_counts[(-1, -1,-1)] = sentence_starts
    ### END YOUR CODE

    # ### YOUR CODE HERE
    # start_word = dataset[0][0]
    # for sentence in dataset:
    #     older = last = None
    #
    #     for current in sentence:
    #         unigram_counts[current] = unigram_counts.get(current, 0) + 1
    #         if last is not None:
    #             bigram_counts[(last, current)] = bigram_counts.get((last, current), 0) + 1
    #         if older is not None:
    #             trigram_counts[(older, last, current)] = trigram_counts.get((older, last, current), 0) + 1
    #
    #         older = last
    #         last = current
    #
    # unigram_counts[start_word] /= 2 # we only use it for the bigram probability
    # token_count = sum(unigram_counts.values()) - unigram_counts[start_word]
    # ### END YOUR CODE

    return trigram_counts, bigram_counts, unigram_counts, token_count

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    prob_log_sum = 0
    token_num = 0

    ### YOUR CODE HERE
    lambda3 = 1-lambda1-lambda2
    for sentence in eval_dataset:
        for idx in range(len(sentence)):

            #unigram prob
            unigram_prob = unigram_counts.get(sentence[idx],0) * 1./ train_token_count

            #bigram prob
            if idx > 0 :
                if sentence[idx - 1] in unigram_counts :
                    bigram_prob = bigram_counts.get((sentence[idx - 1], sentence[idx]),0) *1. / unigram_counts[sentence[idx - 1]]
                else:
                    bigram_prob = 0
            else:
                if -1 in unigram_counts :
                    bigram_prob = bigram_counts.get((-1, sentence[idx]), 0) * 1. / unigram_counts[-1]
                else:
                    bigram_prob = 0

            #trigram prob
            if idx > 1 :
                if (sentence[idx - 2],sentence[idx - 1]) in bigram_counts:
                    trigram_prob = trigram_counts.get((sentence[idx-2],sentence[idx-1], sentence[idx]), 0) * 1. / \
                                   bigram_counts[(sentence[idx - 2],sentence[idx - 1])]
                else:
                    trigram_prob = 0
            elif idx > 0 :
                if (-1, sentence[idx - 1]) in bigram_counts:
                    trigram_prob = trigram_counts.get((-1, sentence[idx - 1], sentence[idx]), 0) * 1. / \
                                   bigram_counts[(-1, sentence[idx - 1])]
                else:
                    trigram_prob = 0
            else:
                if (-1, -1) in bigram_counts:
                    trigram_prob = trigram_counts.get((-1, -1, sentence[idx]), 0) * 1. / \
                                   bigram_counts[(-1, -1)]
                else:
                    trigram_prob = 0




            prob = (lambda1 * trigram_prob) + (lambda2 * bigram_prob) + (lambda3 * unigram_prob)

            # if prob <= 1. / train_token_count :
            #     prob = 0

            prob_log_sum -= np.log2(prob)
            token_num += 1

    perplexity = 2 ** (prob_log_sum / token_num)
    ### END YOUR CODE



    # ### YOUR CODE HERE
    # sum_of_probs = 0
    # test_token_count = 0
    # for sentence in eval_dataset:
    #     older = sentence[0]
    #     last = sentence[1]
    #
    #     for current in sentence[2:]:
    #         # calculating the probability
    #         unigram_prob = float(unigram_counts.get(current, 0)) / train_token_count
    #         if last in unigram_counts:
    #             bigram_prob = float(bigram_counts.get((last, current), 0)) / unigram_counts[last]
    #         else:
    #             bigram_prob = 0
    #         if (older, last) in bigram_counts:
    #             trigram_prob = float(trigram_counts.get((older, last, current), 0)) / bigram_counts[(older, last)]
    #         else:
    #             trigram_prob = 0
    #
    #         # calculating the linear interpolation
    #         prob = lambda1 * trigram_prob + lambda2 * bigram_prob + (1 - lambda1 - lambda2) * unigram_prob
    #         sum_of_probs -= np.log2(prob)
    #
    #         test_token_count += 1
    #         older = last
    #         last = current
    #
    # perplexity = 2 ** (sum_of_probs / test_token_count)
    # ### END YOUR CODE

    return perplexity

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)

    ### YOUR CODE HERE

    perplexity_list = []

    for lambda1 in np.arange(0.0, 1.1, 0.1):
        for lambda2 in np.arange(0.0, 1-lambda1+0.1, 0.1):
            perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
            perplexity_list.append([perplexity , lambda1 , lambda2 , 1-lambda1-lambda2 ])
            print "[l1 = %0.1f , l2 = %0.1f , l3 = %0.1f] #perplexity: %0.1f" %(lambda1,lambda2,1-lambda1-lambda2 , perplexity)

    min_perplexity = min(perplexity_list,key= lambda x : x[0])
    print "MINIMUM: [l1 = %0.1f , l2 = %0.1f , l3 = %0.1f] #perplexity: %0.1f " % (min_perplexity[1],min_perplexity[2],min_perplexity[3],min_perplexity[0])




    ### END YOUR CODE

if __name__ == "__main__":
    test_ngram()