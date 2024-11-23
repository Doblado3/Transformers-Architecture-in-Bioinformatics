#Prueba de modelo de predicción NLP Bi-gram, accuracy no demasiado fiable

import matplotlib.pyplot as plt
import numpy as np
import Tokenizer as tokenizer
import sys



# ========================================
# Create dataset
# ========================================

file_path = "DataSets/small_vocabulary_sentences/eng-300.txt" 

Lang = tokenizer.Tokenizer(file_path)
Lang.read_dataset()
vocab_size = Lang.vocab_size()

print("=================================")
print("Using Bigram")
print("vocabulary size: {}".format(vocab_size))
print("number of sentences: {}".format(Lang.num_texts()))
print("=================================")

# ========================================
# Compute transition matrix
# ========================================

transition_matrix = np.zeros((vocab_size, vocab_size))

for _, sentence in enumerate(Lang.sentences):
    for i in range(len(sentence) - 1):
        idx1 = sentence[i]
        idx2 = sentence[i + 1]
        transition_matrix[idx1][idx2] += 1

s = np.sum(transition_matrix, axis=1)
for i in range(len(s)):
    if s[i] > 0:
        transition_matrix[i] /= s[i]

# ========================================
# Test
# ========================================

if len(s) < 20:
    print("        ", end="")
    for i in range(len(s)):
        print("{:8s}   ".format(Lang.idx2word[i]), end="")
    print("")

    for i in range(len(s)):
        print("{:8s}".format(Lang.idx2word[i]), end="")
        for j in range(len(s)):
            cp = transition_matrix[i][j]
            if cp == 0.0:
                print("0.0        ", end="")
            else:
                print("{:>3f}   ".format(cp), end="")
        print("")

else:
    keys = np.arange(len(s))
    num_candidate = 5
    n = np.minimum(num_candidate, len(s))
    keys = np.random.permutation(keys)[:n]
    words = []
    vals = []

    for key in keys:
        print("Text:{}".format(Lang.texts[key]))

        # Get the index of the second word from the end.
        # e.g. sentences[key][-1] := <eos>, sentences[key][-2] := last word.
        idx = Lang.sentences[key][-3]

        cp = transition_matrix[idx].copy()
        cp[Lang.word2idx['<eos>']] = 0.0
        sorted_list = sorted(cp, reverse=True)

        w = []
        v = []
        for _ in range(num_candidate):
            _max_idx = np.argmax(cp)
            _max_val = np.max(cp)
            w.append(Lang.idx2word[_max_idx])
            v.append(_max_val)

            cp[_max_idx] = 0.0

        words.append(w)
        vals.append(v)

        print("Predicted last word:")
        for i in range(n):
            if v[i] > 0.0:
                print("\t{}\t=> {:>5f}".format(w[i], v[i]))
        print("")

    x = np.arange(len(keys))
    plt.subplots_adjust(wspace=0.4, hspace=1.6)
    for i in x:
        plt.subplot(num_candidate, 1, i + 1)
        plt.title(Lang.texts[keys[i]])
        plt.bar(x, vals[i], tick_label=words[i], align="center")
        plt.ylim(0, 1)
        plt.ylabel("prob.")

    plt.show()