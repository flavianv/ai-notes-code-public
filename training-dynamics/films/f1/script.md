# Transformer Training Dynamics 1: Trainable at step zero

Story spine: a courier of size one travels through a freshly initialized Transformer, and another courier, the gradient, travels back. Seven design choices keep both couriers alive; each chapter fixes one place where a courier can die and ends by asking the question the next chapter answers. Every number comes from `lab.py` (seed 20261014, NumPy, CPU) and is asserted by `make_narration.py`. These are equation checks on random matrices, not trained language models.

---

## 1. the courier

**On screen:** title card. Three 48-layer networks, 256 wide: weights a little too small, exactly right, a little too big. A log plot of the courier's size against layer. The same picture backwards for the gradient. The seven-stop map.

**Voice-over:**
A Transformer, before its first training step, is a pile of random numbers: dozens of matrices filled with noise. Training will turn that noise into something that predicts text. But before it can learn anything, one thing has to be true. A signal has to be able to travel through the whole network and come back.

Meet the courier. It is a number of size one, and it carries the input forward. Here are 48 layers, each 256 wide. In the first network, every weight is a little too small, and each layer shrinks the courier by about 10 percent. After 48 layers it has size 0.004. Nothing arrives. In the second network, the weights are a little too big, 12 percent more per layer, and the courier arrives at size 186. In the third network the weights are exactly right, and the courier arrives at about the size it left with.

The same happens backwards. The gradient is a second courier, travelling the other way, carrying the message: change this weight by this much. If it vanishes, the early layers never learn. If it explodes, the first update wrecks the network.

This film is about step zero: how a Transformer keeps both couriers alive at initialization. There are seven fixes, and every one of them is a design choice you will find in real models. We will derive each fix, show where on the network it lives, and tick it off on a map. You need vectors, matrix multiplication and the chain rule, and nothing else.

**Reader:** "Random matrices. Big dreams."

---

## 2. the route

**On screen:** the route of the courier: token, embedding, L blocks, unembedding, softmax, loss. Seven red markers on the route, numbered, each labelled with the chapter that fixes it.

**Voice-over:**
First, the route. A token becomes a vector. That vector travels through a stack of identical blocks. Each block normalizes the vector, lets attention mix information between positions, and adds the result back. Then it normalizes again, passes each position through a small network called the MLP, and adds that back too. At the top, one more matrix turns the vector into a score for every word, a softmax turns the scores into probabilities, and the loss compares them with the true next word.

Training is everything that happens to the weights in this picture. And along this route there are seven places where a courier can die. One: the matrices themselves, if their scale is wrong. Two: the dot product inside attention. Three: the softmax that follows it, when its scores spread too far apart. Four: those same attention scores, later in training, when they drift out of range. Five: the depth, where small errors multiply. Six: the normalization. Seven: the MLP and its activation function.

We take them in this order, and every chapter starts by showing where on this route we are. The forward pass itself is built step by step in the Intro to Transformers films, so here we only follow the courier.

**Reader:** "Tokens in. Probabilities out."

---

## 3. mean is not magnitude

**On screen:** a number that is plus one or minus one: dots at minus one and plus one, a mean marker at zero. The variance definition. A bar showing that tripling a variable multiplies its variance by nine. The two kinds of average: over initializations, and over tokens.

**Voice-over:**
Stop one starts with a measuring tool. How big is a random number? Take a number that is plus one or minus one, each with probability one half. Its average is zero. But it is never zero: every time you look, it has size one. So the average cannot tell us the size, because the pluses and minuses cancel. We need a measure that does not cancel.

That measure is the variance: the average squared distance from the mean. For our number the mean is zero, so the variance is the average of one squared, which is one. The standard deviation is the square root of the variance, so it is measured in the same units as the number itself.

Two rules make everything else easy. First, if you multiply a variable by a, its variance is multiplied by a squared. Triple the sign, and the values are plus or minus three, so the variance is nine. Second, if you add two independent variables, their variances add.

One habit before we go on: always ask what is random. An average over random initializations is a different thing from an average over the tokens of one trained model. At step zero, the only randomness a model has is its initialization, so that is the average we take.

**Reader:** "Zero on average. Never zero. Like my inbox."

---

## 4. why square roots appear

**On screen:** random walks. A histogram of 20,000 sums of 100 random signs. The expansion of the variance of a sum, with diagonal terms in blue and cross terms in coral. The correlation curve.

**Voice-over:**
Now add many random numbers. Take 100 random signs and sum them. Each step is plus one or minus one, so the sum wanders. Where does it end up? Not at 100. Five random walks are on screen, and none of them gets anywhere near it. Their typical distance from zero is about ten, the square root of 100, as we are about to derive.

Here is why. Write the variance of a sum of d variables as the average of the squared sum. Squaring a sum gives two kinds of terms: each variable times itself, which is its own variance, and each pair of different variables, which is their covariance. If the variables are independent, every covariance is zero. What remains is d times the variance of one variable. So d independent signs add up to variance d, and a typical size of square root of d.

Compare the other extreme. If all 100 signs are the same, they add up to exactly 100. Ten when they are independent, one hundred when they agree. What sits between those is correlation. Give every pair a small correlation rho, and the covariance terms come back: the variance becomes d times one plus d minus one times rho. For d equal to 100 and rho equal to 0.1, that is 1,090. A correlation of just 0.1 turns a typical size of 10 into a typical size of 33.

So independence is doing a lot of work. Random initialization relies on cancellation, and learning creates correlation. Every calculation in this film is a baseline for step zero, not a law about trained models. And a matrix multiplication is made of exactly these sums.

**Reader:** "Independent until proven guilty."

---

## 5. what a random matrix does

**On screen:** a row vector x times a random matrix W gives y, with one output highlighted. The derivation of the variance of y_j. Two histograms of 25,600 outputs: weight variance one against weight variance one over d.

**Voice-over:**
Where are we on the route? Everywhere: every matrix in the network, from the embedding to the last projection, starts as random numbers. We look at one of them.

Take one input vector x with d numbers and a weight matrix W. One output number is y j, the sum over r of x r times W r j. That is d products, added up. At initialization the weights are independent random numbers with mean zero and variance s squared. Now use what we just learned.

The mean of every product is zero, so the output has mean zero. The variance of each product is x r squared times s squared. The products are independent, so their variances add. The variance of y j is s squared times the squared length of x.

Take d equal to 512 and a real input with squared length 536. If the weights have variance one, the output has variance 536, a typical size of 23. Now make the weights smaller: choose s squared equal to one over d. The output variance is 536 over 512, which is 1.05. The output is the same size as the input.

That is the rule: weight variance one over d. Be careful with the words. It is the variance that is one over d, and the standard deviation is one over the square root of d.

Here is how it looks in PyTorch: one line sets the weights to a standard deviation of one over the square root of d. The default of a linear layer is a uniform distribution with variance one over three d, a little smaller.

Go back to the 48-layer network. This one rule is the difference between a courier that shrinks 10 percent per layer and one that arrives at the size it left with. Stop one is ticked off. But this protects the forward direction only. What about the way back?

**Reader:** "Start small. Stay humble."

---

## 6. forward and backward scale

**On screen:** one linear map with arrows both ways; forward sums over d_in, backward sums over d_out. Bars for d_in = 512, d_out = 2048. The ReLU keeps half the second moment. A 20-layer ReLU network under two initializations.

**Voice-over:**
Backpropagation runs through the same matrix in the other direction. A gradient arrives at the output with 2,048 numbers, and the matrix turns it into 512 numbers for the input. Going forward, each output sums over the input width. Going backward, each input gradient sums over the output width. If those two widths differ, one weight variance cannot make both sums come out right.

Take an input width of 512 and an output width of 2,048. With weight variance one over 512, the forward signal keeps its size: a second moment of 1. The gradient sums over 2,048 numbers instead of 512, so it comes out at 4: four times too big. Choose one over 2,048 instead, and the gradient is right at 1, while the forward signal drops to a quarter.

Xavier initialization splits the difference. It uses a variance of two over the sum of the widths, 0.00078 here. The forward signal comes out at 0.4 and the gradient at 1.6. Neither is exactly right, but neither is off by a factor of four.

A ReLU changes the calculation. It sets every negative number to zero, so for a symmetric input it keeps only half the second moment: 1 goes in, one half comes out. To compensate, He initialization doubles the variance, to two over the input width. Without that doubling, a ReLU network shrinks the courier by a factor of root two at every layer.

In PyTorch, these are the Xavier and Kaiming initializers, and they read the widths from the weight shape. The code is on screen.

Each part of a Transformer, the GELU activation, gated MLPs, residual branches and tied embeddings, needs its own version of this calculation. The principle stays the same: match the sum you are actually computing. Stop one is done, for real. Now the courier reaches attention.

**Reader:** "Half the signal? ReLU, we need to talk."

---

## 7. the attention scaling

**On screen:** a query and a key, d_k coordinates each. The derivation line by line. A table of the measured variances for four widths. The softmax row: 64 keys at width 64, raw against scaled.

**Voice-over:**
Where are we on the route? Inside every block, in the attention layer, at the dot product between a query and a key. There the courier meets a dot product. Each query is compared with each key by multiplying them coordinate by coordinate and adding up: q dot k. This is another sum of d k products, so we can compute its size. Let the query and key coordinates be independent, centered, with variance one.

Each product q r times k r has mean zero. Its second moment factors into two second moments, one times one, so each product has variance one. If the products are uncorrelated across coordinates, the variances add, and the dot product has variance d k. Divide by the square root of d k, and the variance drops back to one.

This holds at every width: the raw variance is 16, 64, 256 or 1,024, and after dividing by the square root it is one each time. If instead you divide by d k itself, the variance is one over d k, and it shrinks as the head gets wider.

Why does the size matter? Because the next step is a softmax. Take one query and 64 keys at width 64. With raw scores, the top key takes 79 percent of the weight, and the row is nearly one-hot. With the scaled scores, the top key gets 11 percent, and the weights are spread out. At width 1,024 the raw row puts 95 percent on one key. In PyTorch, the scaled dot-product attention function divides by the square root of the head width by default, and the code is on screen. This is the second ticked stop. And the question it leaves is: the proof assumed queries and keys are independent. What happens when training changes that?

**Reader:** "Divide by d? Now nobody stands out."

---

## 8. what training changes

**On screen:** two arrows q and k, with the cosine. The mixing experiment: k = a·q + noise, with the mean score and variance. A softmax row card: the same scores shifted by 5 and multiplied by 3.

**Voice-over:**
A trained model has no obligation to keep queries and keys independent. In fact it should not. Matching a query to a key is exactly what attention is for. The dot product is the length of q times the length of k times the cosine of the angle between them. At initialization the cosine is near zero. After training, a query and the key it should find point the same way.

Take the extreme case: the key equal to the query. A query dotted with itself is its squared length, which is about 64, and dividing by the square root of 64 leaves 8. So even with the square-root scaling, an aligned pair scores eight against about zero for the others.

Softmax only reacts to differences within a row. Add 5 to every score and the weights do not change at all: the scores 1, 2.2, 0.4, 1.5 give weights 0.15, 0.51, 0.08 and 0.25 either way. Multiply every score by 3 and the row sharpens to 0.02, 0.87, 0.004 and 0.11. Its entropy, a measure of how spread out the row is, falls from 1.19 to 0.47.

So the square root fixes the scale at step zero, and the model is free to move away from it. What we should watch is the spread of the scores within a row, and how well queries and keys align, rather than one global average. And a sharp row has a price, which is the next stop.

**Reader:** "Independent at birth. Aligned by experience."

---

## 9. softmax and saturation

**On screen:** the softmax formula; the two-key curve p(1 − p); the gap between scores against the sensitivity; the cross-entropy example. The max-subtraction code.

**Voice-over:**
Stop three starts with a tool we will use from here on: the Jacobian. A function f takes a vector x with n numbers and returns a vector y with m numbers. Its Jacobian is the m by n table of all the partial derivatives. Row i, column j says how much output i changes when input j is nudged: J i j is the partial derivative of y i with respect to x j. Near a point, the function behaves like this table: a small nudge delta changes the output by J times delta.

An example. Let y one equal x one squared plus x two, and y two equal x one times x two. The table is 2 x one, 1 in the first row, and x two, x one in the second. At x equal to 1 and 3, it is 2, 1 and 3, 1. Nudge x one by 0.1, and y one moves by about 0.2 and y two by about 0.3. For a linear layer, y equals W times x, the Jacobian is W itself.

Now the part that matters for training. The Jacobian of two functions in a row is the product of their Jacobians: that is the chain rule. And the gradient travels backward through the transposed table: if the gradient at the output is g, the gradient at the input is J transpose times g. Take W equal to 1, 2; 0, 1, and an output gradient of 1, 1. The input gradient is W transpose times that, which is 1, 3. Backpropagation is exactly this: one transposed Jacobian per layer. It also shows why depth matters, because a product of many tables can grow or shrink, which we will see two chapters from now.

Where are we on the route? Still inside attention, one step after the dot product: the softmax that turns the scores of a row into weights. Now the softmax. It turns a row of scores into positive weights that sum to one: the weight of key j is e to the power z j, divided by the sum of those over all keys. Now ask the question training asks: if I nudge one score, how much does the weight move?

With two keys, the answer is p times one minus p. At an even split, that is one quarter. When one key already has weight 0.999, it is 0.001. That happens when the scores differ by about seven. So when a row is nearly one-hot, the gradient through it is a thousand times smaller, and a mistake in that row is hard to fix. This is saturation. With many keys, the Jacobian is the diagonal of the weights minus the weights times their transpose, which couples every key in the row. For two keys with weights one half each, it is 0.25, minus 0.25; minus 0.25, 0.25: every row sums to zero, because adding to all scores changes nothing.

In code, softmax subtracts the row maximum before exponentiating. It gives the same weights, because adding a constant changes nothing, but it cannot overflow.

Sharp attention is not always a bug: a learned lookup may need exactly that. The output softmax is different. Followed directly by the cross-entropy loss, its gradient is the probabilities minus the target. Take scores 4, 0, 0 with the true class being the second one. The model puts 96 percent on the first class, and the gradient on that logit is 0.96: a confident, wrong prediction gets a large push. In PyTorch, softmax and the cross-entropy loss are single calls, and even the Jacobian has a function. The code is on screen. So the output layer corrects confident mistakes, and the attention rows do not. Who protects an attention row, later in training, when its scores start to grow?

**Reader:** "Too sure to learn. We all know someone."

---

## 10. keeping attention scores in range

**On screen:** the largest attention score against the growth of query and key norms, with and without QK-norm. The soft-cap curve and its gradient.

**Voice-over:**
Where are we on the route? At the same place as the last two chapters, the attention scores, between the dot product and the softmax. But now much later in training. Stop four. Here is the weak spot we noticed: during training, query and key norms can grow, scores grow with them, and attention rows saturate, which is exactly where the gradient dies. The square root only fixes the scale at step zero. Modern models add a guard that keeps working.

Look at the arithmetic. A score is bilinear: scale the queries and the keys by 10 each, and the score grows by 100. A largest score of about 4 becomes about 400.

QK-norm normalizes each query and key before the dot product, so a score becomes a learned scale times a cosine. A cosine cannot exceed one, so with a scale of 10 the score cannot exceed 10, however large the vectors grow. GLM-4.5 uses it, in its words, to stabilize the range of attention logits.

Soft-capping is the gentler route. Pass each score through a scaled tanh: the cap times tanh of the score over the cap. A score of 200 with a cap of 50 becomes 49.97. Its gradient fades smoothly, to 0.07 at a score of 100, instead of collapsing. Gemma 2 caps its attention scores at 50.

Here is how both look in PyTorch. The code is on screen. In film two we will meet a third option that lives inside the optimizer. For now, the scores are guarded, and the next stop is not inside one layer any more, but across the whole stack of blocks.

**Reader:** "Bounded scores. Calmer gradients."

---

## 11. depth and residual connections

**On screen:** a stack of Jacobians; 1.1 and 0.9 per layer over 48 layers, on a log plot. The residual update and its Jacobian. The variance of the stream against layer for two branch scalings.

**Voice-over:**
Where are we on the route? Not inside one layer any more, but across the whole stack of blocks. Stop five is about depth. A plain deep network multiplies one Jacobian per layer, and the gradient is the product of all of them. If each layer amplifies by 10 percent, then after 48 layers the gradient is 97 times larger. If each shrinks by 10 percent, it is 0.006 of its original size. Small errors compound, which is the courier problem again, in the form of a product.

The residual connection changes the layer. Instead of replacing the vector, a block adds its output: x at the next layer is x plus alpha times f of x. The Jacobian of that layer is the identity plus alpha times the Jacobian of f. The identity gives the gradient a direct path through every layer. It does not guarantee a stable product, though: the branch sizes and correlations still matter.

Look at one coordinate. After the addition, its variance is the old variance, plus alpha squared times the branch variance, plus twice their covariance. In an idealized model where the branches are uncorrelated, each of L branches adds its variance, so the stream's variance grows linearly with depth. With alpha equal to one, the variance after 12, 24 and 48 layers is 13, 25 and 49: one plus the number of layers.

Now scale each branch by one over the square root of L. The added variance becomes L times one over L, which is one, so the total stays bounded. The total is 2, for any depth. It is the same square root as before, one level up. In PyTorch, it is one multiplication on the branch, and the code is on screen.

**Reader:** "A highway for gradients. Still has traffic."

---

## 12. where normalization acts

**On screen:** layer norm and RMS norm on a four-number example. Pre-norm and post-norm block diagrams with the backward path traced. Gradient per block for the two placements, from the lab.

**Voice-over:**
Where are we on the route? At the two norm layers that sit in every block, one before attention and one before the MLP. Stop six. Layer normalization takes one token's vector, subtracts its mean, and divides by its standard deviation, then applies a learned gain and offset. Take the vector 3, minus 1, 0, 4. Its mean is 1.5 and its variance is 4.25, so the normalized vector is 0.73, minus 1.21, minus 0.73 and 1.21. RMS normalization skips the centering and only divides by the root mean square, 2.55 here, which gives 1.18, minus 0.39, 0 and 1.57.

Where the norm sits changes the path of the backward courier. In a pre-norm block, the norm sits on the branch: the residual stream goes around it untouched. In a post-norm block, the norm is applied to the sum, so every gradient passes through it.

We measured the difference on 24 residual blocks at initialization. In the post-norm stack the gradient with respect to the weights is nearly the same in every block: 0.097 in the first, 0.095 in the last. In the pre-norm stack, early blocks get a larger gradient, 0.72 in the first block and 0.17 in the last, because the identity path collects contributions from every later block. Neither stack vanishes or explodes.

So this toy does not show that one placement wins. What it shows is that placement changes the route the gradient takes. The practical preference for pre-norm comes from experiments on full Transformers: Xiong and colleagues found that post-norm models need a learning-rate warmup, while pre-norm models can train without it. That is a result about trained models, which our toy does not reproduce. Also, pre-norm does not keep the stream at unit size, and normalizing a layer's input does not make every query coordinate zero-mean. In PyTorch, a pre-norm block is two lines, and the code is on screen.

**Reader:** "Before or after? The eternal question."

---

## 13. the MLP as a key-value memory

**On screen:** the MLP as a sum of hidden units, each with a key and a value. The parameter count at width 768. The gradients per unit. A grid of units that fire or stay silent for one token.

**Voice-over:**
Where are we on the route? In the second half of every block: the MLP. Stop seven. It takes a vector, multiplies it by a matrix that expands the width by four, applies a nonlinearity, and multiplies by a second matrix that shrinks it back. At width 768, attention has four d squared weights, which is 2.36 million. The MLP has eight d squared, which is 4.72 million: two thirds of the block.

For training, the useful way to read the MLP is as a memory. Each hidden unit is one entry. It has a key, a column of the first matrix, which says what the entry detects, and a value, a row of the second matrix, which says what the entry writes. The token is the query.

Here is a tiny memory with four entries. The token x is 1, 2, 0. Each key is compared with the token by a dot product: the four scores are 2, minus 1, 0.5 and 3. The nonlinearity decides how strongly each entry fires. With a ReLU, the second entry, with a negative score, stays silent, and the others fire with strengths 2, 0.5 and 3. Each firing entry then adds its value to the output, scaled by its strength. With values 1, 0; 0, 1; 1, 1; and 0, 2, the output is 2 times the first value, plus 0.5 times the third, plus 3 times the fourth: 2.5 and 6.5.

So the MLP is a lookup table with soft matching. The token asks, every key answers how well it matches, and the matching entries write their values into the stream. Change a key, and you change what the entry responds to. Change a value, and you change what it says. In PyTorch, it is two linear layers and a ReLU, and the code is on screen.

Now the gradient. A unit's value is pushed by its activation times the error signal arriving at the output: it learns to write what would have helped when its pattern appeared. Its key is pushed toward the inputs that made it useful, multiplied by the slope of the activation. And with a ReLU, a unit that does not fire has slope zero, so both of its gradients are exactly zero.

How many do not fire? At width 768, 50.0 percent of the hidden units are active on a random token. In a small example with 128 units, 74 are silent for the token we tried, and their gradient is exactly zero. So about half the MLP learns from each example.

That is how knowledge accumulates in these layers, and research on trained models agrees: Geva and colleagues found that MLP layers behave like key-value memories, and Meng and colleagues found factual recall localized in mid-layer MLPs.

**Reader:** "If you see this, write that. Times four thousand."

---

## 14. activation functions

**On screen:** the five activation curves and their derivatives. The best-case gradient through 48 sigmoid layers. The gated MLP and its width. Squared ReLU.

**Voice-over:**
Where are we on the route? Still in the MLP, at the nonlinearity between its two matrices. The activation decides which features survive, and which gradients reach the matrix before it. Without one, two matrices in a row collapse into a single matrix. So what should it look like?

Start with the simplest choice, a hard threshold: output one if the input is above zero, and zero otherwise. It is a clean on or off decision. But its derivative is zero everywhere except at the jump, so backpropagation cannot learn through it.

Sigmoid makes the decision smooth. It rises from zero to one, and its derivative, the slope, is at most one quarter. In both tails the slope vanishes. Even in the best case, 48 layers of sigmoid multiply the gradient by one quarter 48 times, and 0.25 to the power 48 is about 10 to the minus 29. Tanh is the same shape centered at zero, with slope one at the origin, but it saturates in both tails too.

ReLU changes the bargain. The positive side is the identity, with slope one, and it is cheap. The negative side is exactly zero, so a unit that is negative on every example stops learning, as we saw. Leaky ReLU gives the negative side a small slope, and PReLU learns that slope.

GELU and SiLU keep the idea of a gate but make it smooth. SiLU is the input times its own sigmoid: z times sigma of z. Large positive inputs pass almost unchanged, large negative inputs are squashed toward zero, and in between there is a smooth transition. GELU is the same idea with the Gaussian probability instead of the sigmoid.

Now the next idea: learn the content and the gate separately. A gated MLP computes two projections of the token. One, passed through SiLU, is the gate. The other is the content. Their product, entry by entry, goes through the output matrix. With SiLU, this is called SwiGLU.

Here is one entry, step by step. Say its gate score is 1.5 and its content is 2. SiLU of 1.5 is 1.23, so the gate is open by a factor of 1.23, and the entry writes 1.23 times 2, which is 2.45. For another entry, the gate score is minus 2 and the content is 3. SiLU of minus 2 is minus 0.24, so the entry writes minus 0.72. A gate can be nearly closed, open, larger than one, or even slightly negative, and that is the freedom a plain ReLU does not have.

The price is a third matrix. To keep the parameter count equal to a four-times-wide MLP, the gated hidden width is eight thirds of d. At width 768 that is 2,048, and three times 768 times 2,048 equals two times 768 times 3,072. Shazeer's matched-budget experiments favored GEGLU and SwiGLU on perplexity, which is a result on one setup, not a proof that gates always win. In PyTorch, it is one line, and the code is on screen.

One more point on the map is squared ReLU: the ReLU times itself. At z equal to 0.5 its value is 0.25, at 2 it is 4, and its slope is two z. It is sparse, but it does not bring dead units back to life.

**Reader:** "State the assumptions. Follow the update."

---

## 15. derive it yourself

**On screen:** the map with all seven stops ticked, a short line under each. Three questions, each with an answer that flips in. The code card.

**Voice-over:**
So that is step zero. The couriers survive because of seven choices. Weights with variance one over d, or two over d for ReLU. Scores divided by the square root of d k. A softmax that is not driven into saturation. Attention scores that are guarded later in training. Residual branches scaled so that variance stays bounded. A norm placed on the branch or on the sum, each changing the gradient's route. And MLP units that can still learn, through an activation whose gradient survives.

Pause and try three questions. Why does a ReLU unit that never fires stop learning? What goes wrong if you divide the scores by d instead of its square root? And why scale each of L residual branches by one over the square root of L?

The answers are on screen. The lab that produced every number in this film is in the code repository, so you can rerun them. And a model that starts trainable still has to be trained. The next film takes one training step, from the loss all the way to the update.

**Reader:** "Step zero: survived."

---

## 16. key references

**On screen:** the references slide: authors, year, title, what each gave this film.

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
