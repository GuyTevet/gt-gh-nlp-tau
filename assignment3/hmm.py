from data import *
import time
from submitters_details import get_details
from tester import verify_hmm_model
import numpy as np


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE

    for sentence in sents:
        for idx , couple in enumerate(sentence):

            total_tokens += 1

            if idx > 1 :
                prev_tag = sentence[idx-1][1]
                prev_prev_tag = sentence[idx - 2][1]
            elif idx > 0:
                prev_tag = sentence[idx-1][1]
                prev_prev_tag = '*'
            else:
                prev_tag = '*'
                prev_prev_tag = '*'

            word = couple[0]
            tag = couple[1]

            q_uni_counts[tag] = q_uni_counts.get(tag, 0) + 1
            q_bi_counts[(prev_tag, tag)] = q_bi_counts.get((prev_tag, tag), 0) + 1
            q_tri_counts[(prev_prev_tag, prev_tag, tag)] = q_tri_counts.get((prev_prev_tag, prev_tag, tag), 0) + 1
            e_word_tag_counts[(word,tag)] = e_word_tag_counts.get((word,tag),0) + 1
            e_tag_counts[tag] = e_tag_counts.get(tag, 0) + 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts


def emission(w,v,e_word_tag_counts,e_tag_counts):
    if (w,v) in e_word_tag_counts:
        return np.log2(e_word_tag_counts[(w,v)] * 1. / e_tag_counts[v])
    else:
        return -np.inf

def transition(t,u,v, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts , lambda1, lambda2):
    lambda3 = 1 - lambda1 - lambda2
    if v in q_uni_counts:
        return np.log2( \
               lambda1 * q_tri_counts.get((t,u,v), 0) / q_bi_counts.get((t,u), 1) + \
               lambda2 * q_bi_counts.get((u,v), 0) / q_uni_counts.get((u), 1) + \
               lambda3 * q_uni_counts.get(v, 0) / total_tokens )
    else:
        return -np.inf


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))

    ### YOUR CODE HERE
    n = len(sent)

    #dynamic programing matrixes (implemented as dicts):
    score_dict = {}
    back_pointers_dict = {}

    #base case
    score_dict[(-1,'*','*')] = 0 #log(1)=0

    #fill matrix (dict)
    for k in range(n):
        w = sent[k]

        #determine optional tags for t,u,v
        if k < 1 :
            u_optional_tags = ['*']
        else:
            u_optional_tags = [tag for tag in q_uni_counts]

        if k < 2 :
            t_optional_tags = ['*']
        else:
            t_optional_tags = [tag for tag in q_uni_counts]

        v_optional_tags = [tag for tag in q_uni_counts]

        for u in u_optional_tags:

            t_list = [t for t in t_optional_tags if score_dict.get((k - 1, t, u), -np.inf) > -np.inf]

            if len(t_list) == 0:
                continue

            for v in v_optional_tags:
                score_list = [score_dict[(k-1,t,u)] + \
                                        transition(t,u,v, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts , lambda1, lambda2) +\
                                        emission(w,v,e_word_tag_counts,e_tag_counts) \
                                        for t in t_optional_tags if score_dict.get((k-1,t,u),-np.inf) > -np.inf]

                idx_of_max = np.argmax(score_list)
                back_pointers_dict[(k,u,v)] = t_list[idx_of_max]
                score_dict[(k,u,v)] = score_list[idx_of_max]

    #find 2 last tags
    max_score = -np.inf
    u_pred = -1
    v_pred = -1

    for (k,u,v) in score_dict:

        if k == n-1 and score_dict[(k,u,v)] > max_score:
            max_score = score_dict[(k,u,v)]
            u_pred = u
            v_pred = v

    predicted_tags[n-1] = v_pred
    predicted_tags[n-2] = u_pred

    #find all other tags using back_pointers_dict
    for idx in reversed(range(n-2)):
        predicted_tags[idx] = back_pointers_dict[(idx+2,predicted_tags[idx+1],predicted_tags[idx+2])]
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    acc_count = 0
    total_words = 0


    #FIXME
    lambda1 = 0.4
    lambda2 = 0.5

    for sentence in test_data:

        sentence_words = [couple[0] for couple in sentence]
        real_tags = [couple[1] for couple in sentence]
        predicted_tags = hmm_viterbi(sentence_words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2)

        total_words += len(sentence)

        for idx in range(len(sentence)):
            got_hmm_error = False
            if real_tags[idx] == predicted_tags[idx]:
                acc_count += 1
            else:
                got_hmm_error = True

        if got_hmm_error:
            with open('hmm_error_tracker.txt','a') as file:
                file.write('real_tags:\n')
                file.write(str(real_tags) + '\n')
                file.write('predicted_tags:\n')
                file.write(str(predicted_tags) + '\n')
                file.write('sentence_words:\n')
                file.write(str(sentence_words) + '\n\n\n')



    return "%0.4f" % (acc_count * 1. / total_words)
    ### END YOUR CODE

    return str(acc_viterbi)

if __name__ == "__main__":
    print (get_details())
    start_time = time.time()
    train_sents = read_conll_pos_file("data/Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("data/Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    verify_hmm_model(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "Dev: Accuracy of Viterbi hmm: " + acc_viterbi

    train_dev_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_time - start_time) + " seconds"

    if os.path.exists("data/Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("data/Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "Test: Accuracy of Viterbi hmm: " + acc_viterbi
        full_flow_end = time.time()
        print "Full flow elapsed: " + str(full_flow_end - start_time) + " seconds"