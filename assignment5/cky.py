from PCFG import PCFG
import math
import numpy as np
import collections

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents

def gen_cky_tree(tokens, bp, i, j, symbol):
    if i == j:
        return "(" + symbol + " " + tokens[i - 1] + ")"
    left, right, split_idx = bp[i][j][symbol]

    return "(" + symbol + " " + gen_cky_tree(tokens, bp, i, split_idx, left) + " " \
                        + gen_cky_tree(tokens, bp, split_idx + 1, j, right) + ")"

def cky(pcfg, sent):
    ### YOUR CODE HERE
    tokens = sent.split(' ') #split sent to word tokens
    pi = collections.defaultdict(lambda: collections.defaultdict(dict))
    bp = collections.defaultdict(lambda: collections.defaultdict(dict))
    for i in range(1,len(tokens)+1):
        for symbol in pcfg._rules.keys():
            found_rule = False
            for rule in pcfg._rules[symbol]:
                if rule[0][0] == tokens[i-1] and pcfg.is_preterminal(rule[0]):
                    pi[i][i][symbol] = np.log(rule[1] / pcfg._sums[symbol])
                    found_rule = True
                    break
            if not found_rule:
                pi[i][i][symbol] = -np.inf

    for l in range(1,len(tokens)):
        for i in range(1,len(tokens)-l+1):
            j = i + l
            for x in pcfg._rules.keys():
                max_val = -np.inf
                max_idx = None
                for rhs in pcfg._rules[x]:
                    if not pcfg.is_preterminal(rhs[0]):
                        y, z = rhs[0]
                        for s in range(i, j):
                            prob = np.log(rhs[1]/pcfg._sums[x])
                            val = prob + pi.get(i,{}).get(s,{}).get(y,-np.inf) + pi.get(s+1,{}).get(j,{}).get(z,-np.inf)
                            if val > max_val:
                                max_val = val
                                max_idx = (y,z,s)
                pi[i][j][x] = max_val
                bp[i][j][x] = max_idx

    #generate tree
    if pi[1][len(tokens)]["ROOT"] != -np.inf:
        return gen_cky_tree(tokens, bp, 1, len(tokens), "ROOT")
    ### END YOUR CODE
    return "FAILED TO PARSE!"

if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
