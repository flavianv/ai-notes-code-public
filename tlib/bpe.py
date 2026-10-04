# Byte-pair encoding, trained by repeatedly merging the most common adjacent pair.
import collections


def bpe_train(word_counts, n_merges):
    words = {tuple(w): c for w, c in word_counts.items()}               # each word starts as a tuple of characters
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] += c                                         # how often does each adjacent pair occur?
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))
        new = {}
        for w, c in words.items():                                       # replace the pair by one new symbol
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == (a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] = c
        words = new
    return merges


def bpe_encode(word, merges):
    w = list(word)
    for a, b in merges:                                                  # apply the merges in the order they were learned
        i = 0
        while i < len(w) - 1:
            if (w[i], w[i + 1]) == (a, b):
                w[i:i + 2] = [a + b]
            else:
                i += 1
    return w


