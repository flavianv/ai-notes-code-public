# mu-P: why your learning rate stops transferring when you widen the model

Standalone part 15. Experiment in `lab.py` (pure NumPy); numbers in `results.json`.

---

## 1. the question

**Voice-over:**
You tune the learning rate of a small model, and it works. You make the model wider, keep the same learning rate, and training blows up. Or it does not blow up, but the best learning rate has quietly moved. This film shows why, with exact numbers, and then shows a recipe, called mu-P, that makes the best learning rate stop moving.

You need three things. A learning rate eta scales every update. The width is the number of units in a layer. And in the standard way of building a network, the initial weights have a variance of one over the number of inputs of their layer.

**Reader:** "Same learning rate. Wider model. Bold."

---

## 2. one step on a wide layer

**Voice-over:**
Here is the intuition. A wider layer has more weights, and every one of them moves a little at each step. If each weight moves by the same amount, the total effect on the output adds up over all the weights. So the output moves more when the layer is wider.

Now we name the pieces. Take one linear layer that computes y as the sum, over the width n, of an input x i times a weight w i. The input has coordinates of order one. The gradient of the loss with respect to y is g. The learning rate is eta.

Now the math, with the numbers plugged in. One step of gradient descent changes each weight by minus eta times x i times g. The new output changes by minus eta times g times the sum of x i squared. If the coordinates of x are plus or minus one, that sum is n, so the output changes by minus eta times n times g.

Take eta equal to 0.01 and g equal to 1. At widths 64, 128, 256, 512, 1,024 and 2,048, the output changes by 0.64, 1.28, 2.56, 5.12, 10.24 and 20.48. Double the width, and the same learning rate makes twice the change. Yet at initialization, with weights of variance one over n, the output has a standard deviation of about 1 at every width. Initialization is stable in width, and the update is not.

Here is what it looks like. A bar for each width: the initial output stays at the same height, and the change after one step grows by a factor of 32 from the narrowest to the widest.

**Reader:** "Stable at the start. Not after one step."

---

## 3. the recipe

**Voice-over:**
Here is the intuition. The fix is to scale the learning rate of each layer, and the initialization of the output layer, so that one step changes the output by an amount that does not depend on the width. The recipe that does it is called the maximal update parameterization, mu-P, introduced by Yang and colleagues.

Now we name the pieces. Take a network with one hidden layer: f of x equals v dotted with tanh of W x. W is the input layer, with n units and a fixed number of inputs. v is the output layer, with n weights. In the standard parameterization, both layers use the same learning rate eta, and v starts with variance one over n.

Now the math, with the numbers plugged in. Look at each layer's contribution to the change of the output. The output layer: its gradient is the error times the hidden activations, whose squared length is of order n. So one step changes the output by about eta times n times the error. To keep that fixed, scale the learning rate of the output layer by one over n. The input layer: its effect passes through the output weights. With variance one over n, their squared sum is about 1, and the change does not grow with n. With the smaller variance one over n squared that mu-P uses for the output layer, the squared sum is one over n, and the change shrinks like one over n, so the input layer's learning rate has to grow like n.

So mu-P is: the input layer has weights of variance one over the number of inputs and a learning rate eta times n. The output layer has weights of variance one over n squared and a learning rate eta over n. Then one step changes the output by an amount that does not depend on the width. Under this recipe, the activations and the updates keep a fixed size, and the network keeps learning features at every width.

Here is what it looks like. A small table of the two recipes, layer by layer.

**Reader:** "Scale each layer by what it sums over."

---

## 4. does the learning rate transfer?

**Voice-over:**
Now the test. We train the one-hidden-layer network on a fixed regression task, with plain gradient descent, at widths 64, 128, 256, 512, 1,024 and 2,048. For each width and each of 13 learning rates, from 2 to the minus 8 up to 16, we record the final loss, as the mean of three runs.

First the standard parameterization. The best learning rate falls as the model gets wider: 0.25 at widths 64 and 128, then 0.125, 0.0625, 0.0625 and finally 0.031 at width 2,048. That is eight times smaller for a model 32 times wider. The learning rate that is best at width 64 makes training diverge at 512 and above.

Now mu-P. The loss curves of all six widths lie on top of each other. At a learning rate of 0.5, the loss is between 0.002 and 0.004 at every width. And at a learning rate of 2, it is below 10 to the minus 4 at every width. The best learning rate found on the narrowest model works on the widest one.

Here is what it looks like. Loss against learning rate, one curve per width. On the left, standard: the curves fan out, and each minimum sits at a different learning rate. On the right, mu-P: one curve.

**Reader:** "Tune small. Train big."

---

## 5. in code, and what to remember

**Voice-over:**
And here is the same thing in PyTorch. Mu-P is two parameter groups with different learning rates, and a different initialization for the output layer.

Some limits. This toy is a network with one hidden layer, trained with plain gradient descent. For Transformers trained with Adam the learning-rate rules differ, and a full mu-P also changes how attention scores and the output are scaled. You do not get it by setting every variance to one over the width. And transfer holds across width, within a family of models: it does not tell you how to choose the depth, the data or the schedule.

To remember: initialization can be stable in the width while the update is not. A step changes the output by about the learning rate times the width. Mu-P scales the learning rate and the initialization of each layer by what it sums over, so that one step changes the output by an amount that does not depend on the width. Then a learning rate tuned on a small model can be used on a large one. Next, the film on grokking asks why a network that can fit anything ends up learning the rule.

**Reader:** "Initialization is not update scale."

---

## 6. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
