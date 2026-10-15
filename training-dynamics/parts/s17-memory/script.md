# Attention as memory: from softmax to the delta rule

Standalone part 17. Numbers from `lab.py` (one write-and-read-back test for every design; delta-rule identities from `../f3/lab.py`).

---

## 1. the question

**Voice-over:**
A model that reads a million tokens cannot afford to compare every pair of them. Newer architectures turn attention into something cheaper, and the cleanest way to understand them is to see attention as a memory: each token writes a key and a value, and each query reads. This film follows one memory through four designs, using the same test every time. Write some random key and value pairs, then read every key back and measure the error.

You need three things. A key is a vector that says what a token is about. A value is the vector it carries. A query is what a token asks for. A memory stores pairs, and a read returns the value whose key matches the query. Writing a pair into a fixed-size memory is done with the outer product of the key and the value, which is a matrix.

**Reader:** "Write pairs. Read them back."

---

## 2. softmax: keep everything

**Voice-over:**
Here is the intuition. The first memory keeps everything. Every pair is written to a list, called the cache. To read, compare the query with every key and take the average of the values, weighted by how well they match. Nothing is forgotten, and nothing is blended, unless two keys are close.

Now we name the pieces. The cache holds N keys and N values. The read is a softmax over the query-key products, followed by a weighted average of the values. Each new token is compared with every earlier token, so the cost over a whole sequence grows as N squared.

Now the math, with the numbers plugged in. Take 4,096 tokens and a width of 64. Comparing every pair costs about 2.1 billion multiply-adds. The cache holds 524,288 numbers, and it grows with every token. On our test, with 16 to 128 pairs and a width of 64, the read-back error is below 2 times 10 to the minus 8: this memory is exact. But at 128 pairs it stores 16,384 numbers, four times the 4,096 of a fixed memory of 64 by 64.

Here is what it looks like. The bars compare the number of stored numbers: the cache grows with the number of pairs, while a fixed matrix stays at 4,096. Exactness has a price, and the price grows with the context.

**Reader:** "Exact, but it never stops growing."

---

## 3. linear attention: add everything

**Voice-over:**
Here is the intuition. Instead of keeping the list, add every pair into one matrix of fixed size, called S. A write is the outer product of the key and the value, and a read is the query times S. The memory never grows, however long the sequence.

Now we name the pieces. S is a matrix with one row per key dimension and one column per value dimension. The write is: S becomes S plus k times v transpose. The read is S transpose times q. Real linear attention also divides by a running normalizer, and uses a feature map on the queries and keys so that the sums factor.

Now the math, with the numbers plugged in. The recurrence is exactly the pairwise kernel attention, computed in another order: on 12 tokens of width 8, the two differ by less than 10 to the minus 15. The state is 72 numbers for width 8, and does not grow with the sequence. At 4,096 tokens and width 64 the recurrent form takes 34 million multiply-adds against 2.1 billion for the pairwise form: 64 times fewer.

Now the flaw. Write the key 1, 0 with the value 1, 0, and then the same key with the value 0, 1. The memory adds both: reading the key returns 1, 1, a blend of the two. On our test, the error is 0.47 at 16 pairs, 0.98 at 64 and 1.41 at 128. An error above 1 is worse than returning zero.

Here is what it looks like. The error of the additive memory climbs with the number of pairs, while the cache stays exact. The fixed state has to pay for its size somewhere.

**Reader:** "Add everything. Blend everything."

---

## 4. the delta rule: write the error

**Voice-over:**
Here is the intuition. A memory needs a way to replace, not only to add. The delta rule reads before it writes. It asks the memory what it returns for this key, compares that with the value we want, and writes only the difference.

Now we name the pieces. The error is the wanted value minus what the memory returns: v minus S transpose k. The write strength beta, between zero and one, says how much of the error to write. The rule is: S becomes S plus beta times k times the error, transposed.

Now the math, with the numbers plugged in. Take the same two writes. After the first, reading the key returns 1, 0. The second wants 0, 1, so the error is minus 1, 1. With beta equal to 1, we write it, and now reading the key returns exactly 0, 1: the old value was replaced, not blended.

Why this is natural: take the loss of the memory on one pair, half the squared distance between S transpose k and v. Its gradient with respect to S is k times that error, so the delta rule is one gradient step on it. Each token trains a tiny linear model inside the forward pass. And it is safe: the matrix that multiplies S, the identity minus beta k k transpose, has largest singular value 1 for a unit key and beta between zero and one, so the state cannot blow up. We measured a maximum of 1.0000.

Training in parallel is harder, because these matrices do not commute. A known identity writes a chunk of steps as one matrix operation, and the chunked form matches the token-by-token recurrence to within 2 times 10 to the minus 15.

On our test, the delta rule gets an error of 0.33 at 16 pairs, 0.72 at 64 and 0.99 at 128, against 0.47, 0.98 and 1.41 for the additive memory. It helps, but a memory of width 64 still cannot hold 128 independent pairs.

**Reader:** "Read first. Then write the error."

---

## 5. gates and hybrids

**Voice-over:**
Here is the intuition. A fixed memory is always full at some point, so it needs to forget. A gate scales the old state down before each write, so old associations fade and make room for new ones. And exact retrieval needs the cache, so some designs keep a few full-attention layers and use the cheap memory everywhere else.

Now we name the pieces. In Gated DeltaNet, a scalar gate alpha multiplies the whole state before the delta update. In Kimi Delta Attention, every key channel has its own gate. A hybrid model mixes layers of both kinds.

Now the math, with the numbers plugged in. With a gate of 0.98, the error is slightly worse up to 64 pairs and slightly better beyond: at 96 pairs, 0.87 against 0.89 for the plain delta rule, and at 128 pairs, 0.93 against 0.99. This test reads every pair back, so it penalizes forgetting by design. It measures capacity, not language-model quality.

Here is what it looks like. Four curves of error against the number of pairs: the cache is flat at zero, the additive memory rises fastest, the delta rule and the gated delta rule rise more slowly. The Kimi Linear paper interleaves Kimi Delta Attention layers with full latent attention layers, three to one, and reports a smaller cache and faster decoding for a context of one million tokens. Reports describe a similar three-to-one mix for Qwen3-Next.

**Reader:** "Forget a little. Keep a few exact layers."

---

## 6. in code, and what to remember

**Voice-over:**
And here is the same thing in PyTorch. The whole delta rule with a gate is four lines in a loop: forget, read, write the error, and read the output.

Some limits. Our test is a toy: random unit keys in a width of 64, which measures how many independent pairs a memory can hold, not how well a model writes. The gate value of 0.98 is one choice. And exact recall at long range is still what the cache is good at, which is why hybrids keep some of it.

To remember: softmax attention is an exact memory that grows. Linear attention is a fixed memory that adds and blends. The delta rule replaces, by writing only the error. Gates let a fixed memory forget, and hybrids keep a few exact layers. Next, the film on scaling laws explains how to spend a training budget between model size and data.

**Reader:** "Exact and growing, or fixed and forgetting."

---

## 7. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
