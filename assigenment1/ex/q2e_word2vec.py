#!/usr/bin/env python

import numpy as np
import random

from q1b_softmax import softmax
from q1e_gradcheck import gradcheck_naive
from q1d_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    x = x / np.expand_dims(np.linalg.norm(x , axis=1),axis=1) #deviding each row by its norm makes each row a unit vector

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    W = outputVectors.shape[0] #extructing dictionary size as W
    D = outputVectors.shape[1] #extructing embadings size as D

    #turn predicted to row vector - if not already
    if predicted.shape[0] != 1 :
        predicted = np.expand_dims(predicted,axis=1)
        predicted = np.transpose(predicted)

    #calc inner product for predicted with all vectors of outputVectors
    inner_prod = np.matmul(predicted,np.transpose(outputVectors)) # dim [1 X W]

    #calc softmax for each word in the dictionary
    s = softmax(inner_prod)

    #caculating the cost
    cost = - np.log(s[0,target]) # since y is a one-hot vector , hence the only element that is not zeroed is y[target]

    #calculating gradPred according to our calculations at 2a
    gradPred_numerator = np.sum((np.tile(np.exp(np.transpose(inner_prod)),(1,D)) * outputVectors),axis=0)
    gradPred_denumerator = np.sum(np.exp(inner_prod))
    gradPred = - np.transpose(outputVectors[target,:]) + gradPred_numerator / gradPred_denumerator # dim [Dx1]

    # calculating grad according to our calculations at 2b
    grad_numerator = np.tile(predicted,(W,1)) * np.tile(np.exp(np.transpose(inner_prod)),(1,D))
    grad_denumerator = gradPred_denumerator
    grad = np.zeros([W,D],dtype=np.float32)
    grad[target,:] = - predicted
    grad += grad_numerator / grad_denumerator

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    W = outputVectors.shape[0] #extructing dictionary size as W
    D = outputVectors.shape[1] #extructing embadings size as D

    #turn predicted to row vector - if not already
    if predicted.shape[0] != 1 :
        predicted = np.expand_dims(predicted,axis=1)
        predicted = np.transpose(predicted)

    #calc inner product for predicted with all vectors of outputVectors
    outputVectorsSampled = outputVectors[indices,:] # dim [ (K+1) x D ]
    inner_prod = np.matmul(predicted,np.transpose(outputVectorsSampled)) # dim [1 X (K+1)]
    neg_samples_sigmoid = sigmoid( - inner_prod[0,1:K+1]) # dim [1 x K]

    # caculating the cost
    cost = -np.log(sigmoid(inner_prod[0,0])) - np.sum(np.log(neg_samples_sigmoid))

    # calculating gradPred according to our calculations at 2c
    gradPred = - (1-neg_samples_sigmoid[0]) * outputVectorsSampled[0,:] - np.sum(outputVectorsSampled[1:K+1,:] * np.tile(np.expand_dims((1- neg_samples_sigmoid),axis=1),(1,D)) , axis=0) # dim [ 1 x D ]

    grad = np.zeros([W,D],dtype=np.float32)# dim [ W x D ]

    grad[indices[0],:] = - predicted * (1 - sigmoid(inner_prod[0,0]))
    grad[indices[1:K+1],:] = np.transpose(np.matmul(np.transpose(predicted) , np.expand_dims((1 - neg_samples_sigmoid),axis=0))) #dim [ K x D ]

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """


    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    center_word_index = tokens[currentWord]

    for out_string in contextWords:
        out_index = tokens[out_string]
        out_cost, out_gradPred, out_grad = word2vecCostAndGradient(inputVectors[center_word_index, :], out_index, outputVectors,
                                                       dataset)
        out_gradIn = np.zeros(gradIn.shape)
        out_gradIn[center_word_index,:] = out_gradPred

        gradIn += out_gradIn
        gradOut += out_grad
        cost += out_cost

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #    skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
