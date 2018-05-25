from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
from submitters_details import get_details
import numpy as np
from copy import copy
from collections import defaultdict

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    #based on Ratnaparkhi (1996)

    #basic
    features['prev_word'] = prev_word
    features['next_word'] = next_word
    features['prevprev_word'] = prevprev_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag

    #combinations
    features['prevprev_prev_tag'] = prevprev_tag + "_" + prev_tag
    features['prev_word_prev_tag'] = prev_word + "_" + prev_tag
    features['prevprev_word_prevprev_tag'] = prevprev_word + "_" + prevprev_tag
    features.update(dict(("prefix_" + str(i), curr_word[:i + 1]) for i in range(min(4, len(curr_word)))))
    features.update(dict(("suffix_" + str(i), curr_word[-i - 1:]) for i in range(min(4, len(curr_word)))))
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greeedy(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    tagged_sent = [(word, None) for word in sent]
    for i in xrange(len(sent)):
        features = extract_features(tagged_sent, i)
        transformed_features = vec.transform(features)
        predicted = logreg.predict(transformed_features)[0]
        predicted_label = index_to_tag_dict[predicted]
        predicted_tags[i] = predicted_label
        tagged_sent[i] = sent[i], predicted_label
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    tags = index_to_tag_dict.values()
    tags.remove('*')  # remove '*' which is invalid tag for the linear regression
    tagged_sent = [(word, '*') for word in sent]
    beam = 50 #reducing compexity by limiting each step to 50 best combinations

    #create inverse dict
    tag_to_index_dict = invert_dict(index_to_tag_dict)
    del tag_to_index_dict['*'] #remove '*' which is invalid tag for the linear regression

    pi = defaultdict(lambda: defaultdict(dict))

    # base case
    pi[0][('*', '*')] = 0  # log(1)=0

    #fill DP matrix
    for i in range(1, len(sent) + 1):
        features = extract_features(tagged_sent, i - 1)

        #find all valid prev tags combinations
        valid_list = list()
        valid_id_dict = dict()
        id = 0

        for (u, w) in pi[i-1]:
            features['prev_tag'] = u
            features['prevprev_tag'] = w
            features['prevprev_prev_tag'] = w + '_' + u
            features['prev_word_prev_tag'] = features['prev_word'] + "_" + u
            features['prevprev_word_prevprev_tag'] = features['prevprev_word'] + "_" + w
            valid_list.append(copy(features))
            valid_id_dict[(u, w)] = id
            id += 1

        valid_vectors = vec.transform(valid_list)
        probs = logreg.predict_log_proba(valid_vectors)

        for v in tags:
            max_prob = -np.inf
            for (u, w) in pi[i - 1]:
                cur_prob = pi[i - 1][(u, w)] + probs[valid_id_dict[(u, w)]][tag_to_index_dict[v]]
                if cur_prob > max_prob:
                    if len(pi[i]) < beam:
                        pi[i][(v, u)] = cur_prob
                    else:
                        min_key = min(pi[i], key=pi[i].get)
                        pi[i].pop(min_key)
                        pi[i][(v, u)] = cur_prob
                    max_prob = cur_prob

    #choose best prob trajectory
    best_v, best_u = max(pi[len(sent)], key=pi[len(sent)].get)
    predicted_tags[len(sent) - 1] = best_v
    predicted_tags[len(sent) - 2] = best_u

    #back tracking
    for k in range(len(sent) - 3, -1, -1):
        predicted_tags[k] = max(pi[k + 2], key=pi[k + 2].get)[1]
    ### END YOUR CODE
    return predicted_tags

def should_add_eval_log(sentene_index):
    if sentene_index > 0 and sentene_index % 10 == 0:
        if sentene_index < 150 or sentene_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    #init log files
    with open('memm_greedy_error_tracker.txt', 'w') as file:
        file.write('')

    with open('memm_viterbi_error_tracker.txt', 'w') as file:
        file.write('')

    total_token = 0

    for i, sen in enumerate(test_data):

        ### YOUR CODE HERE
        ### Make sure to update Viterbi and greedy accuracy

        sent_words = [tup[0] for tup in sen]
        label_tags = [tup[1] for tup in sen]
        greedy_predicted_tags = memm_greeedy(sent_words, logreg, vec, index_to_tag_dict)
        compare_greedy = [(label_tags[i] == greedy_predicted_tags[i]) for i in range(len(sen))]
        acc_greedy += sum(compare_greedy)

        if sum(compare_greedy) < len(sent_words): #error tracked
            with open('memm_greedy_error_tracker.txt', 'a') as file:
                file.write('real_tags:\n')
                file.write(str(label_tags) + '\n')
                file.write('predicted_tags:\n')
                file.write(str(greedy_predicted_tags) + '\n')
                file.write('sentence_words:\n')
                file.write(str(sent_words) + '\n\n\n')


        viterbi_predicted_tags = memm_viterbi(sent_words, logreg, vec, index_to_tag_dict)
        compare_viterbi = [(label_tags[i] == viterbi_predicted_tags[i]) for i in range(len(sen))]
        acc_viterbi += sum(compare_viterbi)

        if sum(compare_viterbi) < len(sent_words): #error tracked
            with open('memm_viterbi_error_tracker.txt', 'a') as file:
                file.write('real_tags:\n')
                file.write(str(label_tags) + '\n')
                file.write('predicted_tags:\n')
                file.write(str(viterbi_predicted_tags) + '\n')
                file.write('sentence_words:\n')
                file.write(str(sent_words) + '\n\n\n')

        total_token += len(compare_greedy)

    acc_greedy = acc_greedy / total_token
    acc_viterbi = acc_viterbi / total_token
        ### END YOUR CODE


    if should_add_eval_log(i):
        if acc_greedy == 0 and acc_viterbi == 0:
            raise NotImplementedError
        eval_end_timer = time.time()
        print str.format("Sentence index: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ", str(i), str(acc_greedy), str(acc_viterbi) , str (eval_end_timer - eval_start_timer))
        eval_start_timer = time.time()

    return str(acc_viterbi), str(acc_greedy)

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    print (get_details())
    train_sents = read_conll_pos_file("data/Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("data/Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    # The log-linear model training.
    # NOTE: this part of the code is just a suggestion! You can change it as you wish!

    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "End training, elapsed " + str(end - start) + " seconds"
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print "Start evaluation on dev set"
    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec, index_to_tag_dict)
    end = time.time()
    print "Dev: Accuracy greedy memm : " + acc_greedy
    print "Dev: Accuracy Viterbi memm : " + acc_viterbi

    print "Evaluation on dev set elapsed: " + str(end - start) + " seconds"
    if os.path.exists('data/Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("data/Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        start = time.time()
        print "Start evaluation on test set"
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec, index_to_tag_dict)
        end = time.time()

        print "Test: Accuracy greedy memm: " + acc_greedy
        print "Test:  Accuracy Viterbi memm: " + acc_viterbi

        print "Evaluation on test set elapsed: " + str(end - start) + " seconds"
        full_flow_end = time.time()
        print "The execution of the full flow elapsed: " + str(full_flow_end - full_flow_start) + " seconds"