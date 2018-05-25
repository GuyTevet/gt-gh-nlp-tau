from data import *
from submitters_details import get_details
import tester

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE

    word_tag_dict = dict() #counts tags for each words
    word_freq_tags = dict() #map words to their most freq tag

    #count tags freq for each word for each tag
    for sentence in train_data:
        for couple in sentence:
            word = couple[0]
            tag = couple[1]

            if word not in word_tag_dict:
                word_tag_dict[word] = {}
                word_tag_dict[word][tag] = 1
            else:
                word_tag_dict[word][tag] = word_tag_dict[word].get(tag,0) + 1

    #find for each word its most freq tag
    for word in word_tag_dict:
        max_word_tag = 0
        for tag in word_tag_dict[word]:
            if word_tag_dict[word][tag] > max_word_tag:
                max_word_tag = word_tag_dict[word][tag]
                word_freq_tags[word] = tag

    return word_freq_tags
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    acc_count = 0
    total_words = 0

    for sentence in test_set:
        for couple in sentence:
            word = couple[0]
            tag = couple[1]

            total_words += 1

            if tag == pred_tags.get(word,'NNP'): #if word not exists in pred_tags we chose to predict 'NNP'
                acc_count += 1

    return "%0.4f" % (acc_count * 1. / total_words)
    ### END YOUR CODE

if __name__ == "__main__":
    print (get_details())
    train_sents = read_conll_pos_file("data/Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("data/Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    tester.verify_most_frequent_model(model)

    if os.path.exists('data/Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("data/Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)



    """
    OUR TEST FOR (1a)
    """
    #print "twoDigitNum - %0d"%model[]
