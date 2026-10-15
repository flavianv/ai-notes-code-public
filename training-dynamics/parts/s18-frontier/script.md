# What changed in the 2026 frontier models? Kimi K3 against the series

Standalone part 18. A dated snapshot (October 2026) built from the Kimi K3, Kimi K2 and Kimi Linear technical reports; every figure is in `facts.json` with its source. Re-record when the models change.

---

## 1. the question

**Voice-over:**
Every part of a Transformer we derived in this series solves one of a few problems: how big the signals are, where the gradient goes, what shape the update takes, and what happens when the model gets bigger. What do the newest models actually change? This film reads one of them, Kimi K3, against that list. It is a snapshot as of October 2026, taken from the technical reports, and it will date. We re-record it when the models change.

You need the four questions. How big are the signals at every layer? Where does the gradient go? What shape does the update take? And what breaks when the model grows: width, data, context length, parameters?

**Reader:** "New parts. Same questions."

---

## 2. signals: scores and depth

**Voice-over:**
Here is the first problem. Attention scores can grow during training until the softmax saturates. In the Kimi K2 report, Muon runs produced logits above a thousand, and QK-Clip, a guard inside the optimizer, capped them at a threshold of 100. K2 trained on 15.5 trillion tokens that way. We derived that guard in the film on Muon.

The second problem is depth. In a residual network every layer adds its output to the same stream, with a fixed weight of one, so the stream is a plain sum of everything before it, and the variance of that sum piles up with depth. Kimi K3 replaces the fixed sum. Each layer has a learned query, and it attends over the outputs of all the layers before it with a softmax, and the weighted sum becomes its input. The layer chooses which earlier results to read. To keep the memory manageable, the report groups the layers into blocks and attends over block-level sums. The report says this improves how information flows across the depth of the model.

This is the same problem we met in the film on depth and normalization. The fix is different: instead of scaling each branch by one over the square root of the depth, the model learns how much of each earlier layer to read.

**Reader:** "The residual sum becomes a lookup."

---

## 3. the update

**Voice-over:**
The second question is the shape of the update. In the film on Muon we saw that a matrix update can be balanced, with every singular value equal, instead of one direction taking most of the step. Kimi K3 is optimized with Muon, applied per attention head: each head's matrices are orthogonalized separately. The report says this per-head orthogonalization improves training stability at larger scales, and that the model keeps the weight-clipping mechanism introduced in Kimi K2.

The point is the same as before. The optimizer is not a detail around the model. It decides which directions of each matrix move, and at this scale it is a stability tool as much as a speed tool.

**Reader:** "Balance the step. Head by head."

---

## 4. scale: context and parameters

**Voice-over:**
The third question is what breaks when the model grows. Two of the four ways to grow matter most here.

The first is the context length. Full softmax attention keeps every key and value, so its cost grows with the square of the length and its cache grows with every token. In the film on attention as memory we followed the alternatives. Kimi K3 uses Kimi Delta Attention, a delta-rule memory with a gate for every channel, in three of every four attention layers, and a full latent attention layer in the fourth. Across its 93 layers that is 69 delta-attention layers and 24 full ones, for a context of 1 million tokens. The Kimi Linear paper, on a smaller model with the same three-to-one mix, reports a smaller cache, up to 75 percent less, and up to 6 times faster decoding at a context of one million tokens.

The second is the number of parameters. Every token pays for every weight, unless only some weights are used. Kimi K3 is a mixture of experts: the feed-forward part of each layer is split into 896 experts, and a router chooses 16 for each token. The model has 2.8 trillion parameters, and uses about 104 billion of them for each token. That is the MLP of the film on what an MLP stores, with a router in front. The report states an improvement of about 2.5 times in overall scaling efficiency over Kimi K2.

**Reader:** "Exact where it counts. Cheap everywhere else."

---

## 5. the answer sheet

**Voice-over:**
Put it together. For each problem we derived, what does the newest model do? Attention scores that grow: a clip inside the optimizer. The residual stream that piles up with depth: a learned lookup over earlier layers. The shape of the update: Muon, per head. The cost of long context: delta-attention layers with a few full-attention layers mixed in. The cost of parameters: a mixture of experts that uses 104 billion of 2.8 trillion weights per token.

Some limits. This is one model, read from its own report, in October 2026. The reports state results; they do not give us the ablations behind each choice. Our derivations explain why each problem exists and what a fix has to do: they do not show that a particular fix is the best one. And the numbers will change with the next model.

The parts are new, but the questions are not: how big are the signals, where does the gradient go, and what shape does the update take.

**Reader:** "New parts. Same questions."

---

## 6. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
