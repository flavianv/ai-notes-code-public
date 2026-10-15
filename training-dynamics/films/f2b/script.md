# Transformer Training Dynamics 2B: Optimization tricks

Story spine: the gradient says which way each weight should move. The update decides how far, how fast, per coordinate and in what shape. We build one update rule piece by piece, from plain gradient descent to Adam to Muon, and finish by putting the whole loop in PyTorch. Every chapter follows the same order: intuition, the pieces named, the math with the numbers plugged in, a plot, and PyTorch. Every number comes from `lab.py` and is asserted by `make_narration.py`.

---

## 1. from gradient to update

**On screen:** title card. The update stage as a pipeline: gradient, clip, momentum, per-coordinate scale or matrix shape, learning rate with a schedule, weight decay, new weights. Seven parts of the update, numbered.

**Voice-over:**
In the last film we followed the penalty backward through every module. Every weight now has a gradient: the direction in which the loss grows fastest. The update stage decides what to do with it.

The simplest answer is to move each weight a little against its gradient. That is gradient descent, and it already works. But there are four questions it does not answer. How far should the step be? Should every coordinate move at the same rate, or should a rarely-updated coordinate be allowed bigger steps? Should we trust one noisy gradient, or an average of the recent ones? And should the update treat a weight matrix as a bag of independent numbers, or as a matrix?

We build the answer one piece at a time. First the size of the step, and why curvature limits it. Then momentum, which averages over time. Then RMSProp and Adam, which scale every coordinate. Then the guards around the step: warmup, clipping and precision. Then schedules and batch size. And finally the matrix-aware updates, Muon and Shampoo, and the guard that keeps attention scores in range.

Each part gets the same treatment: first the intuition, then the pieces named, then the math with the numbers plugged in, then a plot, and last the PyTorch. At the end, the whole loop is one short program.

**Reader:** "How far? How fast? In what shape?"

---

## 2. the step size

**On screen:** the map with the learning rate highlighted. A ball in a bowl with three step sizes; the weight after each step; the stability limit; the edge of stability.

**Voice-over:**
Where are we? At the learning rate, the multiplier between the gradient and the change.

Here is the intuition, with no math. Picture a ball rolling in a bowl, and you move it by looking at the slope under your feet. A small step is safe but slow. A big step may jump past the bottom, to the other side. A bigger one jumps higher than where it started, and every jump is bigger than the last. Steeper bowls, meaning larger curvature, forgive less: the steeper the bowl, the smaller the step must be.

Now we name the pieces. The learning rate is eta. The curvature, written h, is how quickly the slope changes. The loss is the bowl, a half h w squared. The weight after the step is w minus eta times the gradient.

Now the math, with the numbers plugged in. Take h equal to 2 and start at w equal to 1. The gradient is h times w, which is 2. With eta equal to 0.25, the step is 0.25 times 2, which is 0.5, and the new weight is 1 minus 0.5, which is 0.5. Every step multiplies the weight by one minus eta h, here one minus 0.5, which is 0.5. The weight goes 1, 0.5, 0.25, 0.125. With eta equal to 0.75, eta h is 1.5 and the multiplier is minus 0.5: 1, minus 0.5, 0.25, minus 0.125. It jumps across the bottom each time, but the jumps shrink. With eta equal to 1.1, eta h is 2.2 and the multiplier is minus 1.2: 1, minus 1.2, 1.44, minus 1.73. Each jump is bigger than the last. So the step is stable when eta h is below 2, that is, when eta is below two over h.

Here is what it looks like. The three paths: the first slides in, the second zigzags in, the third flies out. And real training does not stay safely below this limit. We trained a small tanh network with full-batch gradient descent and a learning rate of 0.4. The limit for the sharpness, the largest curvature of the loss, is two over eta, which is 5. The sharpness starts at 2.77, rises, and then hovers at 4.95, right at the limit, while the loss keeps falling with 11 sudden jumps. This is the edge of stability, reported by Cohen and colleagues: training sits at the largest step the curvature allows.

And here is the same thing in PyTorch. The optimizer is one line, and its step is the update we computed by hand. Now, a plain step uses one gradient. What if we average several?

**Reader:** "Too big a step. Flew out of the bowl."

---

## 3. momentum

**On screen:** the map with momentum highlighted. A heavy ball; steady and noisy gradients; the running sum; the momentum paths.

**Voice-over:**
Where are we? Right after the gradient, before the learning rate: the gradient is replaced by an average of recent gradients.

Here is the intuition. Think of a heavy ball rolling downhill. If the slope keeps pointing the same way, the ball speeds up. If the slope flips back and forth, the pushes cancel and the ball barely moves. Momentum gives the update a memory: consistent directions add up, and noisy directions cancel.

Now we name the pieces. The momentum buffer v keeps a running sum of past gradients, each older one counted a little less. The decay, mu, is typically 0.9. And the update uses v in place of the gradient.

Now the math, with the numbers plugged in. The rule is v equals mu times the old v, plus the new gradient, and then the weight moves by eta times v. Take mu equal to 0.9 and a gradient that is 1 every step. The first v is 1. The second is 0.9 times 1, plus 1, which is 1.9. The third is 0.9 times 1.9, plus 1, which is 2.71. Then 3.44, 4.10, 4.69, climbing towards one over one minus mu, which is 10. A steady direction is amplified ten times.

Now a noisy gradient that alternates plus 1, minus 1. The first v is 1. The second is 0.9 times 1, minus 1, which is minus 0.1. The third is 0.9 times minus 0.1, plus 1, which is 0.91. Then minus 0.18, 0.84, minus 0.25. It stays below 1: the noise does not accumulate.

Here is what it looks like. The two buffers over six steps: the steady one grows, the noisy one stays small. In effect, momentum raises the learning rate along directions the gradient agrees with, and leaves it alone along directions it disagrees with.

And here is the same thing in PyTorch. Momentum is an argument of the optimizer, and the buffer is stored per parameter. Momentum fixed the direction, but not the scale of each coordinate. That is next.

**Reader:** "Ten times faster, if you keep agreeing."

---

## 4. per-coordinate scale: RMSProp

**On screen:** the map with the per-coordinate scale highlighted. Two coordinates with gradients 100 and 0.01; their step sizes with and without scaling.

**Voice-over:**
Where are we? After the gradient, before the learning rate, on every single coordinate.

Here is the intuition. Imagine two weights. One always gets huge gradients, the other tiny ones. With one learning rate for both, the first one jumps wildly and the second one barely moves. RMSProp gives every weight its own volume knob: it keeps track of how large that weight's gradients usually are, and divides by it. Large gradients are turned down, small ones are turned up, and every weight takes steps of about the same size.

Now we name the pieces. v is a running average of each coordinate's squared gradient. Its square root is the typical size of that coordinate's gradient. The update divides each gradient by that typical size. Epsilon is a tiny number that stops a division by zero.

Now the math, with the numbers plugged in. The rule is v equals beta times the old v, plus one minus beta times the gradient squared, and the step is eta times the gradient over the square root of v plus epsilon. After many steps with steady gradients, v settles at the gradient squared. Take two coordinates, with gradients 100 and 0.01. Their v values settle at 10,000 and 0.0001, and the square roots are 100 and 0.01. The RMSProp step is eta times 100 over 100, which is eta, and eta times 0.01 over 0.01, which is also eta. With plain gradient descent, the steps would be eta times 100 and eta times 0.01: ten thousand times apart.

Here is what it looks like. On a bowl that is 100 times steeper in one direction, loss equals a half of 100 x squared plus y squared, plain gradient descent must keep its learning rate below 0.02 or it blows up in the steep direction. So it crawls along the shallow one: after 120 steps its loss is still 4.5. Momentum reaches 0.003. Adam, which we meet next, reaches 0.0002.

And here is the same thing in PyTorch. RMSProp is one optimizer, and its state is the running average of the squared gradients. Be careful with one thing: this second moment is not a Hessian estimate, and rescaling coordinates one at a time is not invariant to rotating the parameter basis. Next, Adam.

**Reader:** "Every coordinate gets its own volume knob."

---

## 5. Adam and AdamW

**On screen:** the map with Adam highlighted. The two running averages; the bias at the start; the corrected first step; decoupled weight decay.

**Voice-over:**
Where are we? At the same place as the last two chapters, with both ideas combined.

Here is the intuition. Adam is momentum and RMSProp together: an average of the gradient for the direction, and an average of the squared gradient for the scale. There is one repair. Both averages start at zero, so at the beginning they are too small, and dividing by a too-small scale makes the first steps too big. Adam divides that bias out.

Now we name the pieces. m is the running average of the gradient, with decay beta one equal to 0.9. v is the running average of the squared gradient, with decay beta two equal to 0.999. The hats are the bias-corrected versions: m hat is m divided by one minus beta one to the t, and v hat is v divided by one minus beta two to the t. AdamW adds weight decay, applied directly to the weights, separately from the gradient.

Now the math, with the numbers plugged in. Say the first gradient is 1. Then m is 0.9 times 0 plus 0.1 times 1, which is 0.1, and v is 0.999 times 0 plus 0.001 times 1, which is 0.001. Without the correction, the step would be 0.1 over the square root of 0.001, which is 3.16. And RMSProp with beta equal to 0.999, which has no momentum term, would take a step of 1 over the square root of 0.001: 31.6 times too big. With the correction, m hat is 0.1 over 0.1, which is 1, and v hat is 0.001 over 0.001, which is 1, so the step is exactly eta.

Say the second gradient is 3. Then m is 0.9 times 0.1 plus 0.1 times 3, which is 0.39, and v is 0.999 times 0.001 plus 0.001 times 9, which is 0.01. The corrections are one minus 0.9 squared, which is 0.19, and one minus 0.999 squared, which is 0.002. So m hat is 2.05, v hat is 5.0, and the step is 2.05 over the square root of 5.0, which is 0.92 eta. Whatever the gradient size, the step stays close to eta.

AdamW applies the decay directly: the weight is multiplied by one minus eta times lambda before the step. With eta equal to 0.001 and lambda equal to 0.1, that is 0.9999. Putting an L two penalty into the loss instead would send the decay through Adam's moments, and give different dynamics.

Here is what it looks like. The first steps, with and without the correction: 31.6 times too big, against exactly one. And a reminder that a learning rate of 3 times ten to the minus 4 means different things in SGD, Adam and Muon: never compare them directly.

And here is the same thing in PyTorch. AdamW takes the two betas and the decay, and the typical language-model setting is on screen. Now, the guards around this step.

**Reader:** "Whatever the gradient size, take a step of about eta."

---

## 6. the guards: warmup, clipping, the update ratio

**On screen:** the map with the guards highlighted. A learning rate that ramps up; a gradient vector clipped to a circle; the update ratio.

**Voice-over:**
Where are we? Around the update: warmup acts on the learning rate, clipping acts on the gradient, and the update ratio is a gauge we read after the step.

Here is the intuition. At the very beginning of training, the weights are random, the activations are changing quickly, and the optimizer's averages are still unreliable. A full-size step now could break things before they settle. Warmup starts with a small learning rate and raises it gently. Clipping is a different guard: when one batch produces an unusually large gradient, it shrinks that gradient to a maximum length, so a single bad batch cannot throw the weights far away. And the update ratio asks, after the step, how big was the change compared with the weights?

Now we name the pieces. T warm is the number of warmup steps. The clip threshold, c, is the maximum allowed norm of the gradient. The update ratio is the size of the step divided by the size of the weights.

Now the math, with the numbers plugged in. Warmup: the learning rate at step t is eta max times the smaller of one and t over T warm. With T warm equal to 50, at step 10 the learning rate is 10 over 50, which is 0.2 of the maximum. Clipping: take a gradient with entries 3 and 4. Its norm is the square root of 9 plus 16, which is 5. With a threshold of 1, it is multiplied by 1 over 5: the clipped gradient is 0.6 and 0.8. With Adam, clipping does not directly bound the final update, because the second-moment division sits in between.

The update ratio: take a 512 by 512 matrix with entries of variance one over 512, so its norm is the square root of 512, which is 22.6. An Adam step of 3 times ten to the minus 4 on each of the 262,144 entries has norm 3 times 10 to the minus 4 times 512, which is 0.154. The ratio is 0.154 over 22.6, which is 0.0068. If it climbs much higher, or falls to zero, something is wrong.

Here is what it looks like. The warmup ramp reaches its maximum at step 50. The gradient (3, 4) is shrunk onto the unit circle along the same direction.

And here is the same thing in PyTorch. A linear warmup scheduler, a clipping call before the optimizer step, and two lines to log the update ratio. Next, the numbers themselves: how many bits we keep.

**Reader:** "Log the step, not just the gradient. Receipts!"

---

## 7. precision

**On screen:** the map with precision highlighted. The fp16 range on a log axis; an underflowing gradient rescued by a scale; the order scale, unscale, clip, step.

**Voice-over:**
Where are we? Upstream of everything: the numbers in which the gradient is computed.

Here is the intuition. To make training fast, the forward and backward passes use numbers with fewer bits. Fewer bits means a smaller range: some numbers are too big to store, and some are too small to tell from zero. A gradient that is too small simply disappears. Loss scaling is a trick: multiply the loss by a big number first, so every gradient is that much bigger and survives, and divide the scale out again before the optimizer sees it.

Now we name the pieces. Half precision, fp16, has a range between about 6 times ten to the minus 8 and 65,504. Bfloat16 keeps the range of the full 32-bit format but only 7 bits of mantissa. The scale, S, is the number the loss is multiplied by.

Now the math, with the numbers plugged in. In fp16, 70,000 is above the maximum 65,504, and becomes infinity. A gradient of 10 to the minus 8 is below the smallest representable number, and becomes exactly zero. Now scale the loss by 1,024. The gradient is 1,024 times 10 to the minus 8, which is 1.025 times 10 to the minus 5: representable. After the backward pass, we divide by 1,024 again: back to about 10 to the minus 8, in full precision, in the optimizer.

Bfloat16 has the opposite trade: with 7 mantissa bits, the spacing of numbers near 1 is 2 to the minus 7, which is 0.0078. So 1 plus 0.004 is rounded back to 1. The range is huge, so no scaling is needed, but small updates can be rounded away, which is why the weights and the optimizer state are kept in higher precision.

There is one rule of order. Suppose the true gradient norm is 0.5 and the scale is 1,024: the scaled gradient has norm 512. If we clipped it at 1 while it was still scaled, we would clip a perfectly good gradient. So the order is: scale, backward, unscale, clip, step.

Here is what it looks like. The fp16 range as a bar on a log axis: 10 to the minus 8 falls below it, 70,000 above it, and the scaled gradient lands inside.

And here is the same thing in PyTorch. Autocast chooses the precision of each operation, the grad scaler multiplies and divides the loss, and the unscale call comes before the clipping. Next, how the learning rate should change over a whole run.

**Reader:** "Unscale before you clip."

---

## 8. schedules

**On screen:** the map with the schedule highlighted. Three learning-rate curves; a noisy bowl; the loss under each.

**Voice-over:**
Where are we? On the learning rate again, but now over the whole run, not just the start.

Here is the intuition. Early on, a large learning rate makes fast progress. But a large learning rate also means noisy steps, and noise keeps the weights bouncing around the bottom of the valley instead of settling in. Shrinking the learning rate at the end lets the weights settle. A good part of what looks like late learning is simply the steps becoming small enough to land in the valley that they were bouncing across.

Now we name the pieces. The schedule is the learning rate as a function of the step. Cosine decay follows half a cosine from the maximum down to nearly zero. Warmup, stable, decay holds the maximum for most of the run, then decays quickly over the last part.

Now the math, with the numbers plugged in. Cosine: eta t equals a half of eta max times one plus the cosine of pi t over T. Take eta max equal to 0.2 and T equal to 400. At step 200, the cosine of pi over 2 is 0, so eta is a half of 0.2 times 1, which is 0.1. At step 300, the cosine of three pi over 4 is minus 0.707, so eta is 0.1 times 0.293, which is 0.029. For warmup, stable, decay with the last 20 percent as decay: at step 360, the learning rate is 0.2 times 40 over 80, which is 0.1.

Here is what it looks like. We tested this on a noisy bowl: curvature 1, unit gradient noise, learning rate 0.2, 20 runs averaged. With a constant rate the loss stalls at 0.053, close to the prediction of 0.056, because every step is jostled by noise of the same size. With a cosine decay it ends at 0.0053. With the flat stretch and a short decay it ends at 0.011. Same model, same data, same average step: the schedule alone makes a ten-fold difference.

And here is the same thing in PyTorch. A warmup scheduler followed by a cosine scheduler, chained, and the scheduler step is called once per optimizer step. Related to the noise is another dial: the batch size.

**Reader:** "Slow down to land."

---

## 9. batch size

**On screen:** the map with the batch highlighted. One gradient as the true gradient plus noise; the noise shrinking with the batch; steps to a target.

**Voice-over:**
Where are we? At the gradient itself: how many examples it is computed from.

Here is the intuition. A minibatch gradient is the true gradient plus noise, because we only looked at a handful of examples. Look at more examples and the noise averages out. Up to a point, doubling the batch lets you take half as many steps for the same progress. Past that point, the gradient is already accurate, and a bigger batch just costs more compute.

Now we name the pieces. B is the batch size. Sigma squared is the variance of the gradient of a single example. The noise floor is the loss level that noise alone keeps you at. The critical batch size is the batch size beyond which more examples stop helping.

Now the math, with the numbers plugged in. The variance of a minibatch gradient is sigma squared over B. On the noisy bowl, the noise floor of the loss is eta sigma squared over 2 B times 2 minus eta h. With eta equal to 0.2, h equal to 1 and sigma equal to 1, that is 0.2 over 2 B times 1.8, which is 0.0556 over B. For B equal to 1, the floor is 0.056. For B equal to 4, it is 0.014. For B equal to 16, it is 0.0035. How many steps does it take to reach a loss of 0.02? With a batch of 1, never: the floor is above the target. With a batch of 4, 11 steps. With 16, 64 or 256, always 8.

Here is what it looks like. Five loss curves for five batch sizes: the large batches all fall together, the small ones level off at their floor. The critical batch size grows as the loss falls, which is one reason large runs increase their batch size during training.

And here is the same thing in PyTorch. When a batch does not fit in memory, we split it into micro-batches, divide each loss by the number of micro-batches, and take one optimizer step at the end: gradient accumulation. We now know the size, the scale, the schedule and the batch. The last question is the shape.

**Reader:** "Noise shrinks like one over B."

---

## 10. matrix updates

**On screen:** the map with the matrix shape highlighted. An input circle stretched into an ellipse; two ways to measure the size of a matrix; singular values of a gradient.

**Voice-over:**
Where are we? In the same place as the per-coordinate scale, but now treating a weight matrix as a matrix, not as a bag of numbers.

Here is the intuition. A weight matrix transforms vectors: it stretches some directions and shrinks others. An update to the matrix changes that stretching. If the update puts nearly all its size into one direction, the other directions barely change: the update is lopsided. Controlling the average size of the update tells you nothing about that. Matrix-aware optimizers ask for the update to be balanced across directions.

Now we name the pieces. The singular values of a matrix are its stretching factors along its principal directions. The Frobenius norm adds up the squares of all of them: it is the total size. The spectral norm is the largest one: the worst-case stretch of any unit input.

Now the math, with the numbers plugged in. Take a gradient with singular values 100, 10 and 0.1. The Frobenius norm is the square root of 100 squared plus 10 squared plus 0.1 squared, which is 100.5. The spectral norm is 100. A plain gradient step keeps those proportions: the update has singular values in the ratio 1, 0.1 and 0.001. Almost all of the change goes into one direction of the representation, and the weak directions barely move.

Here is what it looks like. The circle of unit inputs becomes an ellipse after the update, very long in one direction and almost flat in the others. The bars show the three singular values on a log scale: a factor of one thousand between the first and the last.

And here is the same thing in PyTorch. Both norms and the singular values are one call each. What should the update look like instead? That is the next chapter.

**Reader:** "One direction hogs the update."

---

## 11. Muon

**On screen:** the map with the matrix shape highlighted. The spectral budget; the ideal update; Newton–Schulz iterations; the polynomial.

**Voice-over:**
Where are we? In the update: Muon replaces the per-coordinate scale with a matrix-shaped step.

Here is the intuition. Give every direction the same step. Take the gradient, keep its directions, but set every singular value to the same number. Then no direction is neglected and none dominates. Computing that exactly would need a singular value decomposition at every step, which is slow. Muon approximates it with a few matrix multiplications.

Now we name the pieces. U, Sigma and V transpose are the singular value decomposition of the gradient G. The nuclear norm is the sum of the singular values. The budget, eta, is the largest allowed spectral norm of the update. Newton–Schulz is an iteration that pushes all singular values towards one using only matrix products.

Now the math, with the numbers plugged in. Among all updates with spectral norm at most eta, the one that lowers the linearized loss the most is minus eta times U V transpose: every singular value set to eta. The reason is that the change in the loss is the sum over i of sigma i times u i transpose Delta v i, and each of those terms is at most eta in size, because the spectral norm of Delta is at most eta. Setting Delta to minus eta U V transpose makes every term exactly minus eta. Take our gradient with singular values 100, 10 and 0.1, and eta equal to 0.01. Plain gradient descent, scaled to the same worst-direction budget, has singular values 0.01, 0.001 and 0.00001, and lowers the linearized loss by 100 times 0.01, plus 10 times 0.001, plus 0.1 times 0.00001, which is 1.010. The ideal update uses 0.01 in all three directions: 100 times 0.01, plus 10 times 0.01, plus 0.1 times 0.01, which is 1.101. This is a local calculation, not a benchmark.

Newton–Schulz applies the polynomial s becomes a s plus b s cubed plus c s to the fifth to every singular value, with a equal to 3.4445, b equal to minus 4.7750 and c equal to 2.0315. First scale the gradient so the largest singular value is just below one: 0.995, 0.1 and 0.001. One iteration: for 0.995, that is 3.4445 times 0.995, minus 4.7750 times 0.995 cubed, plus 2.0315 times 0.995 to the fifth, which is 0.70. For 0.1, it is 0.34. For 0.001, it is 0.0034, about 3.4 times bigger. After five iterations, the three values are 0.70, 0.71 and 0.47: the smallest has grown by a factor of 470. The coefficients are tuned for fast approximate balance, not exact convergence to one. A singular value that is exactly zero stays zero.

Here is what it looks like. The polynomial curve lifts small values strongly and pulls the large ones down to around one. And the three singular values over five iterations come together.

And here is the same thing in PyTorch. The whole Newton–Schulz iteration is a handful of lines. In practice, embeddings, output heads and norm vectors stay on AdamW. Compare tuned baselines at equal tokens and equal time before saying one is faster. Muon has an older relative.

**Reader:** "State the assumptions. Follow the update."

---

## 12. Shampoo

**On screen:** the map with the matrix shape highlighted. Row and column statistics; the preconditioned update; the one-step case equal to U V transpose.

**Voice-over:**
Where are we? Still in the matrix-shaped update, with an older optimizer that turns out to be a close relative of Muon.

Here is the intuition. Instead of one scale per number, as in Adam, Shampoo keeps one statistic for the rows of the gradient matrix and one for the columns. It then divides the gradient on both sides by those statistics. Directions that always carry large gradients are turned down, on both the row side and the column side.

Now we name the pieces. L is the sum over steps of G times G transpose: it describes the rows. R is the sum of G transpose times G: it describes the columns. The update multiplies G on the left by L to the minus one quarter and on the right by R to the minus one quarter.

Now the math, with the numbers plugged in. Take one step, so L and R come from a single gradient. If G equals U Sigma V transpose, then L is U Sigma squared U transpose, and its inverse fourth root is U Sigma to the minus one half U transpose. The same holds for R. Multiply everything out and the sigmas cancel, leaving exactly U V transpose.

A small example. Take G equal to the diagonal matrix with 2 and 1. Then L is the diagonal with 4 and 1, and so is R. The inverse fourth root of 4 is 0.707. The update is 0.707 times 2 times 0.707, which is 1, in the first slot, and 1 times 1 times 1, which is 1, in the second. The update is the identity: every singular value is 1. Muon is Shampoo without the running memory.

Here is what it looks like. A 4 by 3 gradient with singular values 9.64, 2.46 and 0.31, and the update, with all three singular values equal to 1. Real Shampoo does accumulate, splits large matrices into blocks, updates the roots less often, and borrows a step size from another optimizer. It is not an inverse of the full Hessian.

And here is the same thing in PyTorch. Two accumulators and two matrix roots. One last guard, for attention.

**Reader:** "Rows and columns. Both get conditioned."

---

## 13. QK-Clip

**On screen:** the map with QK-Clip highlighted. Attention logits before and after the clip; the gamma formula; the shared rotary key.

**Voice-over:**
Where are we? After the step, on the weights themselves: the query and key projections.

Here is the intuition. In film one we saw that attention scores can grow during training until softmax saturates. The Kimi K2 report describes Muon runs in which some logits exceeded a thousand. QK-Clip is a guard inside the optimizer: after each step, if the largest logit of a head is too big, shrink that head's query and key weights, so the logits shrink by exactly the right factor.

Now we name the pieces. S max is the largest logit observed for a head in the forward pass. Tau is the threshold. Gamma is the shrink factor: tau over S max, or 1 if S max is below tau.

Now the math, with the numbers plugged in. Scaling both the query and key projections by the square root of gamma multiplies every dot product by gamma. In our toy head the largest logit is 1,279. With tau equal to 100, gamma is 100 over 1,279, which is 0.078, and its square root is 0.280. After the clip, the largest logit is exactly 100.0, and all logits have been multiplied by gamma. This is feedback control of the weights: the logits come from the forward pass before the update, so it does not guarantee that every next score is bounded.

In multi-head latent attention, the logit has a content part and a rotary part, and the rotary key is shared by all heads. So scale the head's content query and key by the square root of gamma, and scale only that head's rotary query by gamma. Both parts shrink by gamma, and no other head is disturbed. If you scaled the shared key instead, a head that was not clipped would change: in our test, its rotary logits move by 40.3, on logits as large as 72.9.

Here is what it looks like. The distribution of the logits before the clip reaches 1,279; after it, everything sits within 100. The Kimi K2 report trained with tau equal to 100 on 15.5 trillion tokens with no loss spikes. That is evidence for one recipe, not a guarantee.

And here is the same thing in PyTorch. Three lines after the optimizer step. Now we have every piece. Let us put them in one loop.

**Reader:** "State the assumptions. Follow the update."

---

## 14. the whole loop

**On screen:** the map with every part lit. The complete training step as a code listing, with each line tagged with its chapter.

**Voice-over:**
Where are we? Everywhere: here is the whole update, in one loop.

Here is the intuition. A training step is a short sequence. Compute the gradient in reduced precision, with a scaled loss. Unscale it. Clip it. Let the optimizer turn it into an update, with momentum, a per-coordinate scale and weight decay. Multiply by the learning rate that the schedule gives for this step. Optionally let Muon handle the matrices and AdamW the rest. Then apply the guard on the attention weights.

Now we name the order. Every line on screen is one of the pieces we met: the batch and its accumulation, the scaler, the clipping, the optimizer, the scheduler, and the guard. The order matters: unscale before clipping, and clip before the optimizer.

And here is the whole loop in PyTorch. It is about fifteen lines. Everything else in this film was about understanding why each line is there, and what number it controls.

The next film is about what thousands of these steps build.

**Reader:** "Fifteen lines. A lot of ideas."

---

## 15. key references

**On screen:** the references slide.

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
