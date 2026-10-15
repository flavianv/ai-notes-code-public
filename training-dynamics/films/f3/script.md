# Transformer Training Dynamics 3: Thousands of steps

Story spine: run one small model for thousands of steps and watch what it builds: a circuit that forms while the loss barely moves, a network that memorizes and then generalizes, the reason the simple solution wins, and what changes when the architecture, the width or the set of trainable weights changes. Seven stops: matching, structure, circuits, hidden change, generalization, scale, new architectures. Every new number comes from `lab.py` (seed 20261014, NumPy, CPU); the circuit and grokking curves are measured results from the series' public code (A1 circuits and grokking.py, seeds 0–2) and are asserted by `make_narration.py`.

---

## 1. thousands of steps

**On screen:** title card. Films 1 and 2 as two ticked strips; three questions for this film.

**Voice-over:**
Film one made a Transformer trainable: at step zero every signal and gradient has a sensible size. Film two took a single step, from a loss to a change in a weight. Now we run thousands of steps, and ask what comes out.

There are three questions. What do the steps build? Why does the result work on inputs it never saw? And what changes when the architecture, the width, or the set of trainable weights changes?

Along the way we will watch a circuit form while the loss barely notices, train a network that memorizes first and generalizes ten thousand steps later, and see why the simple solution wins. Then we will read linear attention, delta attention and the adaptation methods, LoRA among them, against everything we derived. As before, every new number is computed or measured, and you can rerun it.

**Reader:** "Many steps. Big plot."

---

## 2. matching and message content

**On screen:** the score as a bilinear form; the value-output map; a gauge change that leaves every score unchanged; the scores before and after.

**Voice-over:**
Start with what a head contains after many steps. Multiply the query and key projections together and the score becomes a bilinear form: one token state, a learned matrix W Q times W K transpose, and the other token state. That matrix is the head's notion of a match: where to look.

The value and output projections combine in the same way, into a message map. The attention weights choose which sources contribute, and the value-output map decides what their features write back into the residual stream.

These products are what you should read, and the individual coordinates are not. Insert any invertible matrix R after the queries and its inverse transpose after the keys, and every score stays exactly the same. Here is a six-token example: the scores on screen are the same before and after a random change of basis.

Two more facts. The product W Q times W K transpose is an 8 by 8 matrix, but its rank is at most the head width, 4, so each head can only look for a few directions. And permuting the tokens together with the mask permutes the output the same way: the head has no built-in sense of order, which is why positions are added separately.

Now, how does a pair of matrices like that come to hold something useful?

**Reader:** "Same scores, new basis. Nobody noticed."

---

## 3. how useful structure accumulates

**On screen:** the weight gradient as a sum of outer products; singular values and effective rank; five features in two dimensions.

**Voice-over:**
Take a linear layer with input x and output gradient u. Its weight gradient is their outer product, and over many examples the gradient is a sum of such outer products. With three examples, the sum of three small matrices equals the matrix product X transpose times U exactly. Directions of the input that keep appearing together with the same useful direction of the output keep being rewarded. That is how structure accumulates.

An MLP adds nonlinear feature detection. Attention mixes information across positions and the MLP transforms each position, and they interact, because the MLP changes the features that later heads match on.

A feature does not have to be one neuron. Several sparse features can share dimensions, which is called superposition. Five features packed into two dimensions, 72 degrees apart, can all be read out, but each overlaps its neighbors: the cosine between adjacent ones is 0.31, and between the next ones 0.81 in absolute value. That overlap is the price of the packing.

We can describe the geometry with the effective rank: the exponential of the entropy of the normalized singular values. For our gradient with singular values 100, 10 and 0.1, the entropy is 0.31 and the effective rank is 1.4. It looks like one direction, with a little else. But a number like this does not prove that a human-readable concept has formed. For that, you need behavioral probes and interventions. Here is a case where we can intervene.

**Reader:** "Five features, two dimensions. Cozy."

---

## 4. a circuit can learn a reusable lookup

**On screen:** a token strip A B ... A → B; the two layers of the induction circuit; the measured scores and the ablations.

**Voice-over:**
Here is what composition can build. Suppose the context contains A followed by B, and later A again. A good prediction is B. Matching the second A directly to the first A is not enough: the model needs what came after the first occurrence.

In the classic two-layer induction circuit, an early head writes information about the previous token into the state at B. A later head, at the repeated A, matches that information in the key at B, and reads B out through the value path.

We trained exactly this in an earlier episode: two layers, four heads, width 64, attention only, on repeated random tokens, with three seeds. In seed 0, one layer-one head attends to the previous token 96 percent of the time. All four layer-two heads attend to the token after the earlier copy, at 87 to 90 percent. Accuracy on the repeated half is 99.7 percent.

Now intervene. Switch off the previous-token head, and the accuracy falls to 7 percent. Switch off the induction heads instead, and it falls to 2 percent. So both parts are needed, and the interventions show it. The other two seeds find the same circuit.

The operators work for any tokens in any position. In small attention-only models, induction heads are linked to better in-context prediction with causal evidence. In larger models, they do not explain in-context learning on their own. The next question is when this circuit forms, and whether the loss shows it.

**Reader:** "Seen it before. Saw what came next."

---

## 5. loss curves can hide internal change

**On screen:** the loss of the random half and the repeated half of each sequence; the average; the one-layer control; the checkpoint shelf.

**Voice-over:**
A smooth average loss can hide different behaviors improving at different times. Split the loss of our circuit model into the random half of each sequence and the repeated half.

The random half stays near 4, because it cannot be predicted: the logarithm of the vocabulary is 3.9. The repeated half is still at 3.5 at step 400. Then it falls to 1.6 at step 500, 0.6 at step 600, and 0.05 by step 3,000, as the induction heads form. Average the two halves, and the same event looks much softer: the combined loss only moves from 3.8 to 2.4 over those 200 steps, because the half that cannot improve dilutes it.

A one-layer model, which cannot compose two heads, never makes the drop. Its repeated-half loss stays between 3.5 and 3.8 across the same steps. So the sudden change is not a measurement artifact: it is the circuit.

Grokking is the extreme case, and the next chapters study it. In some small algorithmic tasks, test performance arrives long after training performance is perfect. So keep checkpoints, fixed evaluation sets and consistent probes. A change in attention or rank suggests a hypothesis. An intervention tests it. And there is no universal schedule in which every language model learns the same abilities.

**Reader:** "Smooth outside. Chaos inside."

---

## 6. fitting noise

**On screen:** the modular addition task; train and test accuracy for real labels and shuffled labels; chance level.

**Voice-over:**
From here on we use one small task where everything can be measured: adding two numbers modulo 97. There are 9,409 pairs. The network sees 30 percent of them, 2,822 pairs, and is tested on the other 6,587. It embeds the two numbers, passes them through a 512-unit ReLU layer, and outputs one of 97 answers.

First, a warning shot. Shuffle the training labels, so the answers are pure noise, and train the same network the same way. It still fits the training set: 100 percent of pure noise, memorized by step 100. On the held-out pairs it stays at chance, 1.0 to 1.05 percent in all three seeds, where chance is one in 97, or 1.03 percent. There is nothing to learn, and it learns nothing.

That is a famous result from Zhang and colleagues: networks that generalize on real data can also memorize random labels. So the number of parameters cannot be what explains generalization. Something about the training itself must prefer the rule over the lookup table when a rule exists. Let us put the real labels back.

**Reader:** "It can learn anything. That's the problem."

---

## 7. grokking: memorize first, generalize much later

**On screen:** train and test accuracy against step on a log axis, three seeds; the weight norm; the control without weight decay.

**Voice-over:**
With the real labels, the training accuracy is above 99 percent within 100 steps. On the held-out pairs, though, the network is still guessing. It has memorized. Then, thousands of steps later, the test accuracy climbs and passes 99 percent at step 11,600 in seed 0, 10,600 in seed 1 and 12,000 in seed 2. That is grokking: generalization arriving more than a hundred times later than memorization.

Watch the weight norm on the same axis. It starts at 113 and falls, as the test accuracy rises, to between 62 and 64. Now remove weight decay and change nothing else. The network still memorizes in 100 steps. But in 20,000 steps its test accuracy never rises above 0.05 percent, and its weight norm grows from 124 to between 286 and 291.

Mechanistic work on this exact task found three phases. First the network memorizes. Then, under an unchanged training accuracy, it builds a general algorithm based on Fourier features of the inputs. And finally weight decay cleans away the memorized lookup table, and only then does the test accuracy jump. The memorizing solution needs large weights, and the algorithm is cheaper. Weight decay keeps paying for the cheaper one. This is the first sign that the optimizer has a preference. We can derive one.

**Reader:** "Studied for the test. Then actually learned it."

---

## 8. why the rule wins

**On screen:** gradient descent on an under-determined least-squares problem; the minimum-norm solution; the double descent curve.

**Voice-over:**
When a network can fit the data in countless ways, training still has to pick one. Gradient descent has a preference built in. Take the simplest case: more unknowns than equations. Here is the theorem. Gradient descent started from zero never leaves the row space of the data matrix, and among the solutions that fit exactly, there is only one in the row space: the one with the smallest norm.

We checked it with 50 equations and 200 unknowns. Gradient descent from zero fits the equations exactly, and lands on the minimum-norm solution, with a norm of 0.62. Another exact fit, found by adding a null-space direction, has a norm of 3.06. Same fit, five times the weights.

Weight decay makes that preference explicit, which is what we saw in grokking. The noise and step size of training tend to steer toward flat regions of the loss, though flatness alone does not explain everything.

One more surprise follows. We fit 40 noisy points with random ReLU features and the minimum-norm solution. With 12 features, the median test error is 0.29. It climbs to 47 at 40 features, exactly where the model can just barely interpolate the data, and then falls again to 0.63 at 160 features. That is double descent: bigger models have more ways to fit, and the minimum-norm choice picks a smoother one among them.

**Reader:** "Many answers. It picks the simplest."

---

## 9. width, mu-P and scaling laws

**On screen:** one SGD step on a wide linear layer; output change against width; the mu-P recipe; the power-law picture.

**Voice-over:**
Initialization can keep activations stable at every width, while the training updates still change with width. For one linear layer and one example, an SGD update is the outer product of the input and the output gradient, with a minus sign and the learning rate.

Apply that update to the same input. The change in the output is the learning rate times the squared norm of the input times the gradient. If the input coordinates are of order one, that squared norm equals the width. Double the width, and the same learning rate makes twice the change.

With the weight variance one over the width, the initial output has a standard deviation of 1 at every width. With a learning rate of 0.01, the output change after one step is 0.01 times the width: 2.6, 5.1, 10 and 20 at widths 256, 512, 1,024 and 2,048. That is one, two, four and eight times. Initialization scale is not update scale. How the upstream gradient and the readout scale matters too, so take this as a warning, not a universal learning-rate rule.

The maximal update parameterization, mu-P, scales the initialization, the learning rates and the readout together, so that feature learning stays meaningful as width grows. That is what lets hyperparameters tuned on a small model transfer to a large one, within the right family of models. You do not get it by setting every variance to one over the width.

Scale the whole recipe, and the loss becomes predictable: across many orders of magnitude it falls as a power law in parameters, data and compute. Kaplan and colleagues measured it, and Hoffmann and colleagues found that for a fixed budget, compute-optimal training uses about 20 tokens per parameter.

**Reader:** "Same learning rate, wider model. Bold."

---

## 10. linear attention: factorization and recurrence

**On screen:** standard attention against the factored form; the running state S and normalizer z; the cost comparison.

**Voice-over:**
Now the architectures. Standard attention computes the similarity of every query with every key, and mixes values. With N tokens, that is N squared pairs. Softmax sits between the query-key product and the value product, so we cannot simply move the parentheses and keep the same function.

Linear attention chooses a similarity that factors as the dot product of transformed queries and keys. The query can then move outside the sum. First accumulate the outer products of transformed keys with values into a matrix S, and the transformed keys into a normalizer z. Each query reads S and divides by its dot product with z. For causal attention these are running sums: each token updates S and z, then reads them. This recurrence is exactly the pairwise kernel attention, computed in a different order.

The cost changes. At 4,096 tokens and feature width 64, the pairwise form takes about 2.1 billion multiply-adds, and the recurrent form 34 million: 64 times fewer, which is N divided by m. The state is a fixed 72 numbers for width 8, and does not grow with the sequence.

This does not generally reproduce softmax attention exactly. A finite feature map replaces its kernel, and compressing the whole past into a fixed state can make precise retrieval harder. Linear refers to the scaling with sequence length, not to the absence of nonlinear functions. And the additive memory writes every association without correcting old ones, which brings us to the delta rule.

**Reader:** "Compress the past. Hope nothing important was in it."

---

## 11. delta attention

**On screen:** the additive memory; the delta rule as one gradient step; the matrix form; the chunked form; Gated DeltaNet and Kimi Delta Attention; the capacity plot.

**Voice-over:**
Chapter ten ended with a state matrix S, where each token adds the outer product of its key and its value. Two things go wrong. If a key is written twice with different values, both stay, and a read returns a blend. And the state only grows, so old and new associations interfere. A memory needs a way to replace, not only to add.

The delta rule reads before it writes. Ask the memory what it returns for this key: S transposed times k. Compare it with the value we want. Write only the error, along the key: S becomes S plus beta times k times the error. The scalar beta, between zero and one, is the write strength. Take the loss of the memory on one pair, half the squared distance between S transposed k and v. Its gradient is k times the error, so the delta rule is one gradient step on it. Each token trains a tiny linear model inside the forward pass.

With a unit key and beta equal to one, the write overwrites: afterwards S transposed k equals v exactly, and directions orthogonal to k are untouched. Expand the brackets, and the update is a matrix product: the identity minus beta k k transpose, times S, plus beta k v transpose. For a unit key and beta between zero and one, that matrix has largest singular value 1, so the state cannot blow up.

Training is harder than for additive attention, because these matrices do not commute. The way out is a known identity: a product of such matrices equals the identity minus a sum of rank-one terms, which is computed inside a chunk by a small triangular recurrence. The chunked form reproduces the token-by-token recurrence exactly, also when two chunks are chained.

Gated DeltaNet adds a scalar gate alpha that scales the whole old state before the delta update, so the model can forget. Kimi Delta Attention gives each key channel its own gate. In our toy, a gate that keeps rows 0 to 7 and halves rows 8 to 15 does exactly that, while a scalar gate scales every row.

Now the capacity of a 64-wide state. We write n random associations and read all of them back. At 16 pairs, the relative error is 0.47 for the additive memory and 0.33 for the delta rule. At 64 pairs: 0.98 and 0.72. At 128 pairs, twice the width: 1.41 and 0.99. The delta rule helps, but a 64-wide state cannot store 128 independent pairs. The gated version, with a gate of 0.98, is slightly worse up to 64 pairs and slightly better beyond: 0.87 against 0.89 at 96 pairs. This test reads every pair back, so it penalizes forgetting by design. It measures capacity, not language-model quality.

**Reader:** "Read first. Then write the error."

---

## 12. what the 2026 frontier models change

**On screen:** three model cards; a sorting board that maps each change to a stop in the three films.

**Voice-over:**
What do the newest models actually change? Kimi K3, released in July, is a mixture of experts with 93 layers. Its attention is mostly Kimi Delta Attention, with full latent attention every fourth layer, and it replaces the residual connection itself. GLM-5 activates 40 billion of its 744 billion weights, uses DeepSeek's sparse attention, and is trained with Muon.

Read them against the three films, and almost every change lands on a problem we derived. Attention scores that grow until softmax saturates: GLM-4.5 adds QK-norm, and Kimi K2 clips query-key products inside the optimizer. Without that clip, its maximum logits passed a thousand. With it, they were capped at a hundred, and 15.5 trillion tokens trained without a loss spike.

Depth: residual branches pile up variance. Attention Residuals replace the fixed sum with attention over earlier layers' outputs, so each layer chooses which earlier results to read. The shape of the update: Muon is now the optimizer in the GLM and Kimi recipes. The MLP: it has become a mixture of experts, hundreds of MLPs per layer with a router choosing a few per token, kept balanced by adjusting a bias instead of adding a loss. The cost of long context: delta attention keeps a fixed state in most layers, and full attention in one layer of four.

The parts are new, but the questions are not: how big are the signals, where does the gradient go, and what shape does the update take.

**Reader:** "New parts. Same questions."

---

## 13. adaptation dynamics

**On screen:** the LoRA update; parameter counts; gradients at initialization; rank and error; 4-bit levels; VeRA and DoRA.

**Voice-over:**
One more way the dynamics change: which parameters are allowed to move. LoRA freezes a pretrained weight matrix and learns its correction as the product of two small matrices, B and A, scaled by alpha over r. The constraint is on the rank of the update, not of the base matrix. At width 4,096 and rank 16, one adapter trains 131,072 parameters instead of 16,777,216: 128 times fewer for this layer. That does not mean 128 times less memory or compute for the whole training.

Initialization decides the dynamics. Start A random and B at zero, so the adapter initially changes nothing. B receives a gradient at once, in our test with norm 17.6. A's first gradient is exactly zero, because it contains B. Once B moves, A learns as well. Set both to zero, and neither ever gets a gradient. LoRA+ uses a larger learning rate for B to balance them.

How much rank is enough depends on the update. For a target update whose singular values decay, the best rank-1 approximation leaves a relative error of 0.85, rank 4 leaves 0.47, and rank 8 leaves 0.21. Higher rank expands capacity; it is not a guarantee of higher held-out quality.

QLoRA attacks the other memory problem, the frozen backbone. It stores the base weights in four bits, with levels placed at the quantiles of a normal distribution and block-wise scales. On Gaussian weights with blocks of 64, the 16 quantile levels give a relative squared error of 0.0085, against 0.0100 for 16 evenly spaced levels. The weights are dequantized for the computation.

VeRA freezes shared random A and B and trains only two small scaling vectors: 4,112 parameters here. DoRA splits each weight column into a magnitude and a direction: the normalized direction keeps every column norm exactly equal to its learned magnitude. Frozen base weights do not guarantee preserved behavior, so evaluate the earlier tasks too.

**Reader:** "Freeze the giant. Teach it a small trick."

---

## 14. make the derivations falsifiable

**On screen:** the terminal; the table of checks from the three films, each with its error; the footer on real experiments.

**Voice-over:**
A derivation you cannot test is only a story. Every film in this series is backed by a lab that runs on a laptop CPU, with nothing but NumPy.

The checks fall in three families. Monte Carlo: sample random matrices and compare the measured variance with the formula, which is how the film one lab checks that scaled dot products have variance 1 at every width. Finite differences: nudge one entry up and down and compare the slope with the formula. The attention gradients in film two agree to 5.8 times ten to the minus 10. And exact identities: the one-step Shampoo update equals U V transpose to 2 times ten to the minus 14, the clip scales every logit by gamma to 4 times ten to the minus 14, and in this film the recurrent and chunked delta rules agree to 1.8 times ten to the minus 15.

These are numerical checks of equations, not language-model benchmarks. For a real experiment, change one factor at a time, compare at equal tokens and wall-clock time, tune the baselines, and use several seeds.

**Reader:** "Nudge it up. Nudge it down. Compare. Repeat."

---

## 15. derive it yourself

**On screen:** the three films on one map with their lessons; three questions and answers; the code card.

**Voice-over:**
So that is the whole chain, across three films. Make every signal start at the right size. Take steps whose size and shape the optimizer chooses well. And thousands of those steps build circuits, and, when weight decay and the optimizer's bias allow it, rules that generalize.

Pause and try three questions. Why can't a one-layer model complete A B, then A again, with B? What drives grokking's late jump? And why is the delta rule gradient descent inside the forward pass?

The answers are on screen. Underneath all of them is one skill: state the assumptions, follow the computation, and measure where real training departs from the model. Everything you saw, the labs, the grokking runs, the circuit runs, is in the repository. Thanks for watching.

**Reader:** "Converged. See you next epoch."

---

## 16. key references

**On screen:** the references slides.

**Voice-over:**
The key references are on screen.

First, the papers on circuits and generalization.

Then the papers on scaling and the new attention layers.

Last, the papers on adaptation.

**Reader:** "Read the papers. They are good."
