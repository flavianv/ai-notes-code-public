# Grokking: why the rule wins

Standalone part 14. Recut from film 3; numbers from `results.json` (film 3 lab).

---

## 1. the question

**Voice-over:**
A neural network with enough parameters can memorize anything, including random noise. So why does it so often learn the rule instead? This film follows one small task where everything can be measured: it shows memorizing and generalizing arriving thousands of steps apart, and then derives which solution gradient descent prefers.

You need three things. Training accuracy is the fraction of training examples the model gets right, and test accuracy is the same on examples it never saw. Weight decay pulls every weight a little toward zero at each step. And the weight norm is the overall size of the weights.

**Reader:** "Memorize anything. Learn the rule?"

---

## 2. fitting noise

**Voice-over:**
From here on we use one small task where everything can be measured: adding two numbers modulo 97. There are 9,409 pairs. The network sees 30 percent of them, 2,822 pairs, and is tested on the other 6,587. It embeds the two numbers, passes them through a 512-unit ReLU layer, and outputs one of 97 answers.

First, a warning shot. Shuffle the training labels, so the answers are pure noise, and train the same network the same way. It still fits the training set: 100 percent of pure noise, memorized by step 100. On the held-out pairs it stays at chance, 1.0 to 1.05 percent in all three seeds, where chance is one in 97, or 1.03 percent. There is nothing to learn, and it learns nothing.

That is a famous result from Zhang and colleagues: networks that generalize on real data can also memorize random labels. So the number of parameters cannot be what explains generalization. Something about the training itself must prefer the rule over the lookup table when a rule exists. Let us put the real labels back.

**Reader:** "It can learn anything. That's the problem."

---

## 3. grokking: memorize first, generalize much later

**Voice-over:**
With the real labels, the training accuracy is above 99 percent within 100 steps. On the held-out pairs, though, the network is still guessing. It has memorized. Then, thousands of steps later, the test accuracy climbs and passes 99 percent at step 11,600 in seed 0, 10,600 in seed 1 and 12,000 in seed 2. That is grokking: generalization arriving more than a hundred times later than memorization.

Watch the weight norm on the same axis. It starts at 113 and falls, as the test accuracy rises, to between 62 and 64. Now remove weight decay and change nothing else. The network still memorizes in 100 steps. But in 20,000 steps its test accuracy never rises above 0.05 percent, and its weight norm grows from 124 to between 286 and 291.

Mechanistic work on this exact task found three phases. First the network memorizes. Then, under an unchanged training accuracy, it builds a general algorithm based on Fourier features of the inputs. And finally weight decay cleans away the memorized lookup table, and only then does the test accuracy jump. The memorizing solution needs large weights, and the algorithm is cheaper. Weight decay keeps paying for the cheaper one. This is the first sign that the optimizer has a preference. We can derive one.

**Reader:** "Studied for the test. Then actually learned it."

---

## 4. why the rule wins

**Voice-over:**
When a network can fit the data in countless ways, training still has to pick one. Gradient descent has a preference built in. Take the simplest case: more unknowns than equations. Here is the theorem. Gradient descent started from zero never leaves the row space of the data matrix, and among the solutions that fit exactly, there is only one in the row space: the one with the smallest norm.

We checked it with 50 equations and 200 unknowns. Gradient descent from zero fits the equations exactly, and lands on the minimum-norm solution, with a norm of 0.62. Another exact fit, found by adding a null-space direction, has a norm of 3.06. Same fit, five times the weights.

Weight decay makes that preference explicit, which is what we saw in grokking. The noise and step size of training tend to steer toward flat regions of the loss, though flatness alone does not explain everything.

One more surprise follows. We fit 40 noisy points with random ReLU features and the minimum-norm solution. With 12 features, the median test error is 0.29. It climbs to 47 at 40 features, exactly where the model can just barely interpolate the data, and then falls again to 0.63 at 160 features. That is double descent: bigger models have more ways to fit, and the minimum-norm choice picks a smoother one among them.

**Reader:** "Many answers. It picks the simplest."

---

## 5. what to remember

**Voice-over:**
To remember: a network that generalizes on real data can also memorize random labels, so the number of parameters does not explain generalization. On a task with a rule, memorizing comes first and generalizing thousands of steps later, and weight decay is what pays for the cheaper, general solution. Gradient descent started from zero lands on the minimum-norm solution, and weight decay makes that preference explicit. When a model can just barely interpolate its data, the error can spike, and then falls again as it grows: double descent. Next, the film on the 2026 frontier models reads the newest model against everything in this series.

**Reader:** "Many answers. It picks the simplest."

---

## 6. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
