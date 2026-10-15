# Induction heads, layer by layer: how a transformer learns to copy a pattern

Standalone part 12. Experiment: the A1 induction experiment of the Understanding Transformers series (3 seeds), results in `results.json` (see `lab.py`).

---

## 1. the question

**Voice-over:**
Give a language model the pair A, B somewhere in its prompt, and later show it A again. It will often predict B. How does a transformer learn to do that? In this film we find the circuit that does it, in a model small enough to inspect layer by layer, and then switch its parts off one at a time to see which ones matter.

You need three things. Attention: each position sends out a query, compares it with the keys of the earlier positions, and reads their values. A head is one attention computation, and a layer runs several heads in parallel, each writing into a shared vector at every position. And a head can be switched off by setting its output to zero.

**Reader:** "Seen A, B once. Predict B next time."

---

## 2. the idea: two steps

**Voice-over:**
Here is the intuition. Take the sequence A, B, C, A. What comes next? A good guess is B, because the last time we saw A, B followed. The rule is: find the earlier copy of the current token, and continue it.

A transformer cannot do that in one step. Attention matches a query against keys, and at the final A the query is about A. But the position holding B is not about A: its token is B.

Now we name the pieces. A previous-token head reads the position just before its own and writes "the token before me was this" into its own column. An induction head sends out a query for the current token, and looks for a key of the form "the token before me was this".

Now the steps. In the first layer, a previous-token head at each position writes the token before it into its column. Now the column for B holds two things: I am B, and the token before me was A. In the second layer, an induction head at the final A sends out the query A. It matches the key "the token before me was A" at B's column. The value at that column carries B, and the output writes that B is likely next.

Here is what it looks like. Matching happens in one place and the payload sits in the same column. Two simple read-write steps compose into "find the earlier copy and continue it". Researchers at Anthropic described this circuit in 2021, and showed it in small attention-only models in 2022.

**Reader:** "Layer one labels. Layer two looks it up."

---

## 3. the experiment

**Voice-over:**
Now let us see whether a model finds this on its own. The data is a start token, a run of random tokens, and then the same run again. The first half is pure noise, so the best possible loss there is about 3.9, the log of the 49 possible tokens. Our models sit at about 4.0. The second half is fully predictable, if the model can find the earlier copy.

The model is as small as we can make it: attention only, no MLPs, four heads per layer, and a width of 64. We train two versions, one layer and two layers, with three random seeds each.

A confession. Our first attempt used repeats of a fixed length of 20 tokens. The two-layer model solved it almost at once, but without any induction head: one head simply learned to look 19 positions back. That is a correct solution to the data we gave it, and the wrong algorithm. So we vary the repeat length during training, from 8 to 32 tokens. Now no fixed offset works, and the only general solution is to match content.

**Reader:** "A model learns what the data allows."

---

## 4. what each layer learns

**Voice-over:**
Here is what we measured. The one-layer model never learns the repeat. It gets the next token right only 6 to 10 percent of the time, and its loss on the repeat stays above 3.2. One layer cannot compose two steps.

The two-layer model gets the repeat right 99.7 to 99.8 percent of the time. And look at how it gets there. For the first 400 steps, the loss on the repeat only creeps down, from about 4.1 to between 3.3 and 3.5. Then it falls off a cliff: by step 600 it is between 0.5 and 0.6, and it ends between 0.03 and 0.05. That sudden drop is the circuit forming. Olsson and colleagues saw the same bump in the loss curve and called it a phase change.

Now look at each layer. For every head in layer one, we measure how much of its attention goes to the position just before. In the first run, a single head does this, with 96 percent of its attention. In the other two runs, two heads do it, and the best scores are 94 and 90 percent. For every head in layer two, we measure how much of its attention goes to the token right after the earlier copy. All four heads do it, with 87 to 92 percent. The other heads score near zero on these tests.

The attention maps show it. On a sequence of ten random tokens, repeated, the layer-one head looks one step back at every position. In layer two, at each position of the repeat, the induction head looks at the token that followed the earlier copy.

**Reader:** "One layer can't. Two layers can."

---

## 5. switching parts off

**Voice-over:**
Seeing the pattern is not proof that the model uses it. So we switch heads off and measure again. Here is the first surprise. Switch off one induction head, and accuracy stays at 98 percent. The other three do the same job.

So we switch off groups. All the previous-token heads: accuracy falls from 99.8 to between 5 and 8 percent. All four induction heads: down to about 2 percent. Keep just one induction head, and switch off the other three: accuracy is between 56 and 66 percent. And switching off the other layer-one heads, the ones that do not look back one step, only takes it to around 80 percent.

That is what makes this a circuit and not a coincidence: remove these heads and the behavior goes, remove the others and it mostly stays. It also shows a real difficulty. Training often builds the same mechanism several times over, so switching off one part at a time can hide a part that matters.

**Reader:** "Switch off one: nothing. Switch off the group: gone."

---

## 6. limits, and what to remember

**Voice-over:**
And here is the same thing in PyTorch. Switching a head off is one line: zero its output before the layer writes into the residual stream. The induction score of a head is the attention it puts on the token right after the earlier copy.

Some limits. Training used repeats of at most 32 tokens. At a repeat of 50, accuracy drops to between 14 and 28 percent: even this clean circuit only works within the range it was trained on. Our model has no MLPs, a task with one rule, and tiny vocabulary. Real models contain many more circuits, and finding one on a few examples does not show that it explains what the model does elsewhere.

To remember: a model that must copy a pattern builds two heads in two layers. The first labels each token with its predecessor, the second looks up the label and copies what followed. The loss barely moves until the circuit forms, and then it drops. Single-head ablations can hide a part that matters, because training builds the mechanism more than once. Next, the film on relations and memory looks at what the MLP stores.

**Reader:** "Two layers. Label, then look up."

---

## 7. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
