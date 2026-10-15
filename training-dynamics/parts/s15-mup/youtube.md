# mu-P: why your learning rate stops transferring when you widen the model

## Description

Initialization can be stable in the width while the update is not: one step changes the output by eta times the width (0.64 to 20.48 from width 64 to 2,048). We derive the mu-P recipe layer by layer, and test it: with the standard parameterization the best learning rate falls 8x as the width grows 32x; with mu-P the loss curves of all six widths lie on top of each other.

Every number in the film is computed or measured by a script you can rerun, and each spoken number is asserted against it before the voice-over is generated.

## Chapters
0:00 The question
0:40 One step on a wide layer
2:34 The mu-P recipe
4:36 Does the learning rate transfer?
6:10 In code, limits and what to remember
7:19 Key references

## Key references
- Yang & Hu (2021), Feature Learning in Infinite-Width Neural Networks, ICML
- Yang et al. (2022), Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer, arXiv:2203.03466

## Code

In github.com/flavianv/ai_tutorials, folder transformer-training-dynamics/v2/.

Series: AI Notes · How Transformers Train (standalone parts).
