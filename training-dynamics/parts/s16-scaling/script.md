# Scaling laws: a bigger model or more data?

Standalone part 16. Numbers: `lab.py` (toy fit and closed form), constants from Hoffmann et al. (2022) as reproduced by Besiroglu et al. (2024), and Epoch AI's refit.

---

## 1. the question

**Voice-over:**
You have a fixed amount of computation to spend on training a language model. Do you build a bigger model, or train a smaller one on more data? This film derives the answer, tests it on a small experiment, and compares it with the published rule of thumb: about 20 tokens per parameter.

You need four things. N is the number of parameters. D is the number of training tokens. The loss is how wrong the model is on text it has not seen. And compute, C, is the number of floating-point operations. For a Transformer, each token costs about 2 operations per parameter going forward and about 4 going backward, so C is about 6 times N times D.

**Reader:** "Bigger model, or more data?"

---

## 2. two gaps above a floor

**Voice-over:**
Here is the intuition. Each time you double the model, the loss improves, but by less than the time before. Data behaves the same way. There is a floor that no model reaches, the irreducible loss of the text itself, and two gaps above it: one that shrinks as the model grows, and one that shrinks as the data grows.

Now we name the pieces. E is the floor. A over N to the alpha is the gap from a model that is too small. B over D to the beta is the gap from too little data. Alpha and beta are the rates at which the gaps close. Kaplan and colleagues measured power laws of this kind in 2020, and Hoffmann and colleagues used the two-gap form in 2022.

Now the math, with the numbers plugged in. We trained 6 small character-level models, from 1,045 to 55,385 parameters, each on 4 amounts of data, from 12,800 to 819,200 characters, one pass over the data each. That is 24 runs. The loss goes from 3.63, for the smallest model on the least data, down to 1.97, for the largest model on the most. Fitting the form gives E equal to 1.26, A equal to 10.3, B equal to 19.8, alpha equal to 0.38 and beta equal to 0.27. The typical error of the fit is 0.08.

Check one run. The model with 9,677 parameters, on 204,800 characters. The gap from the model is 10.3 over 9,677 to the 0.38, which is 0.32. The gap from the data is 19.8 over 204,800 to the 0.27, which is 0.73. Adding the floor, 1.26 plus 0.32 plus 0.73 is 2.31. We measured 2.33.

Here is what it looks like. The heat map is the measured loss for all 24 runs: bigger models and more data both lower it. On log axes, each data size gives a curve that bends towards its own floor, and the fitted form follows them.

**Reader:** "A floor, and two gaps."

---

## 3. splitting the budget

**Voice-over:**
Here is the intuition. With a fixed budget, more model means fewer tokens. The best split is where moving a little compute from one to the other buys nothing: the last bit spent on the model closes the same amount of loss as the last bit spent on data.

Now we name the pieces. The budget is C equal to 6 N D, so D is C over 6 N. Put that into the loss and it becomes a function of N alone. We want the N that makes it smallest.

Now the math, with the numbers plugged in. The loss is E plus A N to the minus alpha, plus B times 6 N over C, to the beta. Set the derivative with respect to N to zero. That gives alpha A N to the minus alpha equal to beta B D to the minus beta: the model gap times alpha equals the data gap times beta. Solving for N gives N star equal to alpha A over beta B, to the power one over alpha plus beta, times C over 6, to the power beta over alpha plus beta. And D star is C over 6 N star.

Put in our toy constants. Alpha A is 3.92 and beta B is 5.36. Take a budget of 10 to the 11 operations. The formula gives N star of about 10,900 parameters and D star of about 1.53 million tokens: 140 tokens per parameter. As a check of the algebra, we can use the published constants and search by brute force: over 4,000 model sizes, the lowest loss is at 32.21 billion parameters, and the closed form gives 32.19 billion.

Here is what it looks like. The curve is the loss along the budget line, as a function of the model size, and its minimum sits where the formula says. A different budget moves the minimum, and the ratio of tokens to parameters changes only slowly.

**Reader:** "Spend until the last bit buys the same."

---

## 4. the published rule

**Voice-over:**
Here is the intuition. The rule everyone quotes comes from one training run. Hoffmann and colleagues trained Chinchilla with 70 billion parameters on 1.4 trillion tokens, which is 20 tokens per parameter, and showed it beat a much larger model trained on fewer tokens.

Now we name the pieces. Their parametric fit has E equal to 1.69, A equal to 406.4, B equal to 410.7, alpha equal to 0.34 and beta equal to 0.28. Epoch AI later refit the same data and reported E equal to 1.82, A equal to 482, B equal to 2,085, alpha equal to 0.348 and beta equal to 0.366.

Now the math, with the numbers plugged in. Chinchilla's budget was 5.76 times 10 to the 23 operations. With the original constants, our closed form gives 32 billion parameters and 3 trillion tokens: 93 tokens per parameter, not 20. With Epoch's refit, it gives 72 billion parameters and 1.33 trillion tokens: 18.4 tokens per parameter, close to what Chinchilla actually used. The replication found problems with the original parametric estimates, while the compute-optimal policy itself holds up.

The ratio is not exactly constant. When alpha and beta differ, it drifts slowly with the budget, as C to the power alpha minus beta over alpha plus beta. With Epoch's constants it goes from 24 tokens per parameter at 10 to the 19 operations down to 17 at 10 to the 25.

Here is what it looks like. Tokens per parameter against budget, for the two sets of constants, with the 20 rule as a line. The refit sits close to it. The original constants climb far above it.

**Reader:** "20 is a result, not a law."

---

## 5. limits, and the code

**Voice-over:**
Some limits. Our toy is a small character model, and its optimum, 140 tokens per parameter, is not a statement about Transformers. It also extrapolates: the optimum for a budget of 10 to the 13 needs 22 million tokens, 28 times more data than we trained on. Real fits extrapolate over many orders of magnitude, and small changes in the constants move the answer, as the original constants showed.

The loss on text is also not the goal. If a model will be served to many users, inference cost matters, and then the better choice is a smaller model trained on more tokens than the compute-optimal rule says. And the rule assumes fresh data: repeating the same text many times changes the curve.

And here is the same thing in code. The whole procedure fits in a few lines: a grid over the two exponents, a least-squares solve for the three constants, and the closed form for the split. No deep learning library is needed.

To remember: the loss has a floor and two gaps. The best split under a fixed budget equalizes what the last bit of compute buys. The famous 20 tokens per parameter comes from a fit, and its constants matter. Next, the film on mu-P explains why a learning rate tuned on a small model breaks when you make it wider.

**Reader:** "Fit it. Then check the constants."

---

## 6. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
