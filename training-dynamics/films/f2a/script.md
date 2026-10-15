# Transformer Training Dynamics 2A: Gradients, module by module

Story spine: the loss hands out a penalty at the output, and every module on the way back takes its share and passes the rest on. We follow that penalty backward through eight modules. Each chapter follows the same order: intuition, the pieces named, the math with the numbers plugged in, a plot, and PyTorch. Every number comes from `lab.py` (small worked examples, exact arithmetic) and is asserted by `make_narration.py`.

---

## 1. the penalty

**On screen:** title card. A training step in four stages: forward pass, loss, backward pass, update; this film is the backward pass. The map of modules the penalty meets, in order.

**Voice-over:**
A training step starts with a prediction and ends with a change in the weights. In between, something has to decide which weights were responsible for the mistake. That something is the gradient, and in this film we follow it. Think of it as a penalty. The loss hands out a penalty at the output. Every module on the way back takes its share, in proportion to how much it contributed, and passes the rest further down.

A step has four stages: the forward pass, the loss, the backward pass, and the update. This film is about the third: how the penalty travels backward, module by module. The next film is about the fourth, the update, and the tricks that make it work.

We take the modules in the order the penalty meets them. The loss itself. One linear layer, the building block of everything. The case where one matrix serves many tokens. The embeddings. The residual connections. The norm layers. The MLP. And attention, with its routes. Each module gets the same treatment: first the intuition, then the pieces named, then the math with the actual numbers plugged in, then a plot, and last the PyTorch.

You need vectors, matrix multiplication, and the Jacobian from film one: the table that says how each output changes when an input is nudged.

**Reader:** "Who gets the blame? Let us find out."

---

## 2. the loss

**On screen:** the map, with the loss highlighted. The four candidate words, the probabilities, and the penalty on each score.

**Voice-over:**
Where are we? At the very end of the network, where the prediction meets the truth.

Here is the intuition, with no math. The model has just said how likely each next word is. The truth is one word. The loss is large when the model gave the true word a small probability, and small when it gave it a large one. The penalty it hands back follows one rule: push up the score of the true word, and push down every other word, each in proportion to how much probability it was given. A wrong word that was very likely gets blamed the most.

Now we name the pieces. The scores the network produces for the candidate words are called logits, written z. The softmax turns them into probabilities p. The target e y is a vector with a one at the true word and zeros everywhere else. The loss is minus the log of the probability given to the true word, averaged over the positions that count.

Now the math, with the numbers plugged in. Take the logits 2, 1, 0 and minus 1, for the words bank, water, loan and steep, and say the true next word is bank. First the exponentials: e to the 2 is 7.39, e to the 1 is 2.72, e to the 0 is 1, and e to the minus 1 is 0.37. They add up to 11.48. Dividing each one by 11.48 gives the probabilities 0.64, 0.24, 0.09 and 0.03. The loss is minus the log of the probability of bank: minus the log of 0.64, which is 0.44.

The penalty on the logits is p minus e y. Plug in: 0.64 minus 1, 0.24 minus 0, 0.09 minus 0, and 0.03 minus 0. That is minus 0.36, 0.24, 0.09 and 0.03. When the loss is averaged over N positions that count, every penalty is divided by N. With six positions, the penalty on bank is minus 0.06.

Here is what it looks like. The curve shows the loss against the probability given to the true word. At a probability of 0.02 the loss is almost 4 and the curve is steep; near 1 it is flat. The bars show the probabilities and the penalty. The true word gets a negative penalty, so gradient descent will raise its score, and each wrong word gets a positive one, in proportion to its probability.

And here is the same thing in PyTorch. Cross-entropy takes the logits and the target index, skips the padding, and the backward call produces exactly this penalty, divided by the number of positions that count. That is the penalty at the output. How does it get into the network? Through a linear layer.

**Reader:** "Stable, smooth, and wrong. A classic."

---

## 3. one linear layer

**On screen:** the map with a linear layer highlighted. Three inputs, two outputs, a line for every weight; the penalty travelling back along the lines; the update of every weight.

**Voice-over:**
Where are we? Everywhere. The unembedding, the attention projections and the MLP are all linear layers. We take the smallest one.

Here is the intuition. A linear layer mixes its inputs into outputs, each output a weighted sum of all the inputs. When a penalty arrives at an output, two things must happen. The weights are blamed, in proportion to how large the input was that they multiplied. And the inputs are blamed too, in proportion to how large the weight was that carried them, so that the layer before can pass the penalty on. One penalty at the output fans out to every weight and every input.

Now we name the pieces. The layer is y equals x W, with three inputs and two outputs, so W has three rows and two columns. G y is the penalty arriving at the outputs. G W is the penalty assigned to the weights, and G x is the penalty sent on to the inputs. And eta is the learning rate.

Now the math, with the numbers plugged in. Take x equal to 1, 2 and minus 1, and a weight matrix with rows 0.5 and minus 1; 1 and 0.5; and minus 0.5 and 2. The forward pass: y one is 1 times 0.5, plus 2 times 1, plus minus 1 times minus 0.5, which is 3. And y two is 1 times minus 1, plus 2 times 0.5, plus minus 1 times 2, which is minus 2. Say the penalty arriving at the outputs is 0.5 and minus 1.

The weights get x transpose times G y: every weight is the input it multiplied, times the penalty at its output. The first row is 1 times 0.5 and 1 times minus 1: 0.5 and minus 1. The second row is 2 times each: 1 and minus 2. The third row is minus 1 times each: minus 0.5 and 1.

The inputs get G y times W transpose. Input one is 0.5 times 0.5, plus minus 1 times minus 1, which is 1.25. Input two is 0.5 times 1, plus minus 1 times 0.5, which is 0. Input three is 0.5 times minus 0.5, plus minus 1 times 2, which is minus 2.25.

Now update every weight with a learning rate of 0.1: the new weight is the old weight minus 0.1 times its penalty. The first weight goes from 0.5 to 0.45, the second from minus 1 to minus 0.9, the third from 1 to 0.9, and so on for all six.

Here is what it looks like. The inputs on the left, the outputs on the right, a line for every weight. The penalty enters at the outputs and travels back along the same lines, and on each line it is multiplied by the number at the other end. After the update, the sum of penalty times output falls from 3.5 to 2.75, a drop of 0.75. That is exactly the learning rate, 0.1, times the squared size of the weight penalty, 7.5.

And here is the same thing in PyTorch. Autograd gives the same two matrices as our hand computation. But a real network applies the same matrix to many tokens at once, and that is the next chapter.

**Reader:** "Input times blame. Blame times weight. Memorize it."

---

## 4. one matrix, many tokens

**On screen:** the map with the whole token row highlighted. The sentence "the river bank steep" as four tokens passing through one W; each word's gradient term; the terms summing to one matrix; one update.

**Voice-over:**
Where are we? In every attention layer and every MLP, wherever the same matrix is applied to every token of the sentence.

Here is the intuition. A sentence is a row of tokens, and the model applies the very same matrix W to each one. Think of W as one employee who handles every token's request. When the penalty comes back, every token has its own complaint about what W did to it. W cannot go in different directions at once, so it takes all the complaints, adds them up, and makes one single change.

Now we name the pieces. X has one row per token. The penalty G has one row per token too: the penalty at that token's output. Each token's contribution to the weight penalty is the outer product of its input row and its penalty row. The sum over tokens is X transpose times G.

Now the math, with the numbers plugged in. Take the four tokens the, river, bank and steep, with three features each: the is 1, 0, 0; river is 0, 1, 1; bank is 1, 1, 0; and steep is 0, 0, 1. The penalties at their outputs are 0.1 and minus 0.2 for the, 0.3 and 0.1 for river, minus 0.4 and 0.2 for bank, and 0.2 and 0.2 for steep.

Each token contributes one outer product. The word the contributes 0.1 and minus 0.2 to the first row. River contributes 0.3 and 0.1 to the second and third rows. Bank contributes minus 0.4 and 0.2 to the first and second rows. Steep contributes 0.2 and 0.2 to the third row.

Add them up. First row: 0.1 minus 0.4 is minus 0.3, and minus 0.2 plus 0.2 is 0. Second row: 0.3 minus 0.4 is minus 0.1, and 0.1 plus 0.2 is 0.3. Third row: 0.3 plus 0.2 is 0.5, and 0.1 plus 0.2 is 0.3. That is one matrix, for all four words. Now one update, with a learning rate of 0.5: the weight in the first row and first column goes from 0.5 to 0.5 minus 0.5 times minus 0.3, which is 0.65.

Here is what it looks like. Four words, each with its own small matrix of penalty, stacked on top of each other and adding up to a single matrix. The same W is updated once, for all the words at the same time. Each word pulls W in its own direction, and the update is their sum.

And here is the same thing in PyTorch. The backward call adds the contribution of every token into the same grad tensor, and it equals X transpose times G. One optimizer step then changes the single matrix. But the words themselves come from a table, and that is the next module.

**Reader:** "One matrix. Many jobs. Relatable."

---

## 5. the embeddings

**On screen:** the map with the embedding highlighted. The vocabulary as a table of rows; the sentence "the bank by the river" picking rows; the gradient landing only on those rows.

**Voice-over:**
Where are we? At the very start of the network, where words become vectors.

Here is the intuition. An embedding is a lookup table: one row per word of the vocabulary. In a sentence, only a handful of words are used, so only their rows took part in the computation, and only those rows are blamed. If a word appears twice, its row is blamed twice, and the blame adds up.

Now we name the pieces. E is the embedding matrix, with one row per word in the vocabulary. Its penalty has the same shape. For each position in the sentence, we add that position's penalty to the row of the word that stood there.

Now the math, with the numbers plugged in. The vocabulary is the, bank, river, loan and by. The sentence is the, bank, by, the, river. The penalties arriving at the five positions are 0.1, minus 0.2, 0.3; then minus 0.1, 0.2, 0; then 0.2, 0, minus 0.1; then 0.3, 0.1, 0; and last 0, minus 0.2, 0.1. The word the stands at positions one and four, so its row gets the sum: 0.1 plus 0.3 is 0.4, minus 0.2 plus 0.1 is minus 0.1, and 0.3 plus 0 is 0.3. Bank, river and by each stand once, so their rows get their own penalty. And loan never appeared: its row gets nothing. Four of the five rows change.

Here is what it looks like. The penalty table has one row per vocabulary word. The row of loan is empty, and the row of the is the sum of two positions. In a real model with a vocabulary of fifty thousand words and a sentence of a few hundred tokens, almost every row is empty.

And here is the same thing in PyTorch. The embedding's weight gradient is mostly zero rows, and there is an option to store it as a sparse tensor. One caution: when the input embedding is tied to the output layer, the output side blames every word in the vocabulary a little, so the shared matrix gets a dense penalty. Now the penalty has reached every weight. But the layers are stacked, so how does it cross a whole block without fading? Through the residual connection.

**Reader:** "Only the words that showed up get blamed."

---

## 6. the residual connections

**On screen:** the map with the residual additions highlighted. A block with a straight path and a branch; the penalty copied onto both; twelve layers with and without the straight path.

**Voice-over:**
Where are we? At the two additions in every block, where a branch is added back to the stream.

Here is the intuition. A residual connection gives the signal two ways through a layer: straight through, and through the layer's own computation. On the way back, the penalty takes both. It is copied: one copy goes straight down, untouched, and the other goes through the layer. So even if the layer's own path squeezes the penalty to almost nothing, the straight copy still arrives.

Now we name the pieces. The block computes x out equals x plus f of x. J f is the Jacobian of f. g out is the penalty arriving at the output of the block, and g x is the penalty at its input.

Now the math, with the numbers plugged in. Going backward, g x equals g out, plus J f transpose times g out. Take J f equal to 0.1, 0.3; minus 0.2, 0.1, and g out equal to 1 and 1. J f transpose times g out is: 0.1 times 1 plus minus 0.2 times 1, which is minus 0.1; and 0.3 times 1 plus 0.1 times 1, which is 0.4. So the branch contributes minus 0.1 and 0.4, and the straight copy contributes 1 and 1. The sum is 0.9 and 1.4.

Now stack twelve layers, each with a branch Jacobian of size 0.1. Without residuals, the penalty is multiplied by 0.1 twelve times: 10 to the minus 12, nothing is left. With residuals, every layer multiplies by one plus 0.1, which is 1.1, and 1.1 to the 12 is 3.14.

Here is what it looks like. On a log scale, the plain stack falls straight down to 10 to the minus 12, while the residual stack climbs gently. The straight path is a highway for the penalty. Film one showed the other side of this, that the stream grows with depth, and how to scale the branches to control it.

And here is the same thing in PyTorch. It is one addition, and autograd handles both copies. Next, the norm layers, which sit on the branch of every block.

**Reader:** "A highway for blame."

---

## 7. the norm layers

**On screen:** the map with the norms highlighted. A vector, its normalized version, and the penalty with its radial part removed.

**Voice-over:**
Where are we? At the two norm layers of every block, one before attention and one before the MLP.

Here is the intuition. A normalization layer rescales a vector to a standard size. So if you made the input twice as large, the output would not change at all. Then it makes no sense to blame the size of the input: the penalty has no component along the input direction. The layer keeps only the part of the penalty that changes the direction of the vector.

Now we name the pieces. RMS norm computes y equals x divided by r, where r is the root mean square of x. g is the penalty at y, and g x is the penalty at x.

Now the math, with the numbers plugged in. Going backward, g x equals one over r times g minus y times y dot g over d. The second term removes the part of g that points along y. Take x equal to 3, minus 1, 0 and 4. The mean square is 6.5, so r is 2.55, and y is 1.18, minus 0.39, 0 and 1.57. Say the penalty g is 1, 0, 0 and 1. Then y dot g is 1.18 plus 1.57, which is 2.75, and dividing by d equal to 4 gives 0.69.

Subtract 0.69 times y from g: 1 minus 0.81, 0 plus 0.27, 0, and 1 minus 1.08. That is 0.19, 0.27, 0 and minus 0.08. Divide by r: 0.075, 0.106, 0 and minus 0.030. Check the claim: the dot product of this penalty with x is 0.075 times 3, plus 0.106 times minus 1, plus minus 0.030 times 4, which is zero. The penalty has no component along x.

Here is what it looks like. The vector x, the penalty g that arrived, and the penalty that is passed on: the passed-on one is perpendicular to x. Whatever pushed along x was removed.

And here is the same thing in PyTorch. The norm is a module, and we can check that its input gradient is perpendicular to its input. Now we have the pieces around the two big sublayers. First the MLP.

**Reader:** "Size is not your problem. Direction is."

---

## 8. the MLP

**On screen:** the map with the MLP highlighted. The four-entry memory from film one, with the penalty flowing back to values, strengths, keys and the input.

**Voice-over:**
Where are we? In the second half of every block: the MLP.

Here is the intuition. The MLP is a memory with entries. On the way forward, the entries that matched the token fired and wrote their values. On the way back, only those entries are blamed. An entry that stayed silent has nothing to answer for. A firing entry is blamed twice: for the value it wrote, in proportion to how strongly it fired, and for how strongly it fired, in proportion to how much its value mattered to the penalty.

Now we name the pieces. Each entry has a key, a value, a strength, which is the activation after the nonlinearity, and a slope, the derivative of the nonlinearity at the score. For a ReLU, the slope is one if the entry fired and zero if it did not.

Now the math, with the numbers plugged in. Use the same four-entry memory as in film one. The token x is 1, 2, 0. The scores are 2, minus 1, 0.5 and 3, the strengths are 2, 0, 0.5 and 3, and the output is 2.5 and 6.5. Say the penalty at the output is 1 and minus 1.

Each value is blamed by its strength times the penalty. Entry one: 2 times 1 and minus 1, which is 2 and minus 2. Entry two is silent: nothing. Entry three: 0.5 times 1 and minus 1, which is 0.5 and minus 0.5. Entry four: 3 times 1 and minus 1, which is 3 and minus 3.

Each strength is blamed by its value dotted with the penalty. Value one, 1 and 0, dotted with 1 and minus 1, gives 1. Value two, 0 and 1, gives minus 1. Value three, 1 and 1, gives 0. Value four, 0 and 2, gives minus 2. Multiply by the slope: 1 for the firing entries and 0 for the silent one. The penalty on the scores is 1, 0, 0 and minus 2.

Each key is blamed by the input times the penalty on its score. Key one gets 1 times 1, 2, 0, which is 1, 2, 0. Key four gets minus 2 times 1, 2, 0, which is minus 2, minus 4, 0. The other keys get nothing. And the input is blamed through the keys: 1 times key one, 2, 0, 0, plus minus 2 times key four, 1, 1, 0, which is 0, minus 2, 0.

Here is what it looks like. The same memory picture as in film one, now with the penalty flowing back along each arrow: the silent entry is greyed out and carries nothing.

And here is the same thing in PyTorch. Two matrices and a ReLU, and autograd computes all of this. A mask shows which units were silent for a batch, and those units received no gradient at all. Now the last module, and the most intricate: attention.

**Reader:** "The silent ones are off the hook."

---

## 9. attention

**On screen:** the map with attention highlighted. Three tokens; the attention weights; the five steps of the backward pass with their numbers.

**Voice-over:**
Where are we? In the attention layer of every block, between the norm and the residual addition.

Here is the intuition. Attention builds each output as a weighted average of the values of the tokens it can see. The penalty on an output therefore splits three ways. Some of it goes to the values that were averaged, in proportion to their attention weights. Some of it goes to the attention weights themselves: a source that helped, a weight that should be higher. And that second part flows on, through the softmax, to the scores, and from the scores to the queries and the keys.

Now we name the pieces. For one head, Q, K and V are the queries, keys and values, one row per token. S is the score matrix, Q K transpose over the square root of d k, and A is the softmax of each row of S, with the future masked out. H equals A V is the output, and U is the penalty that arrives at H. We want G V, G A, G S, G Q and G K.

Now the math, with the numbers plugged in. Take three tokens and d k equal to 2. The attention weights are: row one, 1, 0, 0; row two, 0.33 and 0.67, and 0; row three, a third each. The penalty U is 1, 0; 0, 1; and 1 minus 1. Step one: the values get A transpose times U. Value one gets 1 times 1, plus 0.33 times 0, plus 0.33 times 1, which is 1.33 in its first coordinate.

Step two: the weights get U times V transpose. For the third token, with U equal to 1 and minus 1, the penalty on its three weights is 1, minus 1 and 0: the first source helped, the second hurt. Step three is the softmax step: subtract the weighted average of that row. The weighted average is a third times 1, plus a third times minus 1, plus a third times 0, which is 0. So the penalty on the scores is a third times 1, a third times minus 1 and a third times 0: 0.33, minus 0.33 and 0.

Step four: the queries get G S times K over the square root of d k. For the third token: 0.33 times key one, 1, 0, plus minus 0.33 times key two, 0, 1, which is 0.33 and minus 0.33, divided by 1.41: 0.24 and minus 0.24. Step five is the same with transposes for the keys.

Here is what it looks like. The attention weights, the penalty on the scores, and the penalty on the queries and the keys, as tables. Notice that each row of the penalty on the scores adds up to zero, and that the masked entries are exactly zero.

And here is the same thing in PyTorch. These are the five lines, and autograd gives the same numbers. Now look at what the second step means: why does an attention weight go up?

**Reader:** "Checked against finite differences. Trust issues, resolved."

---

## 10. routes

**On screen:** the map with a single route highlighted. Two sources with values 0 and 2; the mixture; the penalty on each score; the bars before and after one step.

**Voice-over:**
Where are we? Inside the attention layer, on a single route: one source token sending its value to one query.

Here is the intuition. A route is one path from a source token to a query. It gains attention if giving it more weight would lower the loss. But a source is never judged alone: it is judged against what the query already receives, the mixture. A source better than the mixture is rewarded, a worse one is punished, and a source exactly as good as the mixture gets no push at all.

Now we name the pieces. A i j is the weight of source j in the row of query i. h i is the mixture, the weighted average of the values. u i is the penalty arriving at the output. The push on the score of source j is A i j times u i times the value of j minus the mixture.

Now the math, with the numbers plugged in. Two sources with values 0 and 2, and equal weights 0.5 and 0.5. The mixture is 0.5 times 0 plus 0.5 times 2, which is 1. We want the output 2, so the loss is a half of the squared error: a half times 1 squared, which is 0.5, and the penalty u is 1 minus 2, which is minus 1. The push on the first score is 0.5 times minus 1 times 0 minus 1, which is plus 0.5. The push on the second is 0.5 times minus 1 times 2 minus 1, which is minus 0.5.

Take one step with a learning rate of 0.1: the first score goes to 0 minus 0.1 times 0.5, which is minus 0.05, and the second to plus 0.05. The weights become 0.475 and 0.525, and the loss falls from 0.5 to 0.451. If both sources had the same value, say 1 and 1, the mixture would be 1, the difference would be zero for both, and the push would be exactly zero.

Here is what it looks like. After 80 steps the weight on the better source has grown from 0.5 to 0.89, and the loss has fallen to 0.022. That is the whole story of attention learning where to look.

And here is the same thing in PyTorch. Four lines: a softmax over the scores, a weighted average, a squared error, and backward. The gradient on the scores is the push we computed. That is every module. Let us put them back together.

**Reader:** "Be better than average. Get noticed."

---

## 11. one block, who gets what

**On screen:** the map with the whole block highlighted. A table: for each module, what receives a penalty and what it computes.

**Voice-over:**
Where are we? Back at the whole picture, with one block in view.

Here is the intuition. The penalty enters at the loss, and it travels backward through the same modules in reverse order. At each module it is split: one part goes to the weights of that module, to be used in the update, and the other part goes to the module's inputs, to be handed to the module before it. Where there is a residual connection, the penalty is also copied along the straight path.

Now we name what each module does with the penalty. The unembedding turns the penalty on the logits into a penalty on its weights and on the final vector. The final norm removes the radial part. The MLP blames the entries that fired. Attention splits the penalty to the values, the weights and then the queries and keys. The residual additions copy it. And the embedding adds it into the rows of the words that were used.

Now the check. We verified every gradient in this film against finite differences: nudge one entry up and down, and compare the slope with the formula. The largest difference over all the modules is below 4 times ten to the minus 10. That is how you know the formulas are right, and how to debug your own.

And here is the same thing in PyTorch. After the backward call, every parameter has its gradient, and a loop prints the size of each. That is the whole gradient half of a step. Each weight now knows which way to move. The next film is about how far, how fast, and in what shape: the update.

**Reader:** "Every weight knows its share."

---

## 12. key references

**On screen:** the references slide.

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
