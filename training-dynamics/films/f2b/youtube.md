# Transformer Training Dynamics 2B: Optimization tricks

## Description

The gradient says which way each weight should move. The update decides how far, how fast, per coordinate and in what shape. We build one update rule piece by piece, with the numbers plugged in and the PyTorch next to the math: step size and the edge of stability, momentum, RMSProp, Adam and AdamW, warmup, clipping, precision and loss scaling, schedules, batch size, Muon, Shampoo and QK-Clip, then the whole training loop.

Every number in the film is computed or measured by a script you can rerun (NumPy, CPU), and each spoken number is asserted against the script before the voice-over is generated.

What is derived or checked:
- The parabola: a step is stable only while eta times h is below 2, and a toy network at the edge of stability
- Momentum: steady directions add up, noisy ones cancel
- RMSProp: a scale for every coordinate
- Adam: bias correction, 31.6x against 1.0 on the first step; AdamW decoupled decay
- Warmup, gradient clipping and the update ratio
- fp16, bf16 and loss scaling; why to unscale before clipping
- Cosine and warmup-stable-decay schedules on a noisy bowl
- Batch size and the noise floor; gradient accumulation
- Frobenius against spectral norm
- Muon: the balanced step and Newton-Schulz
- Shampoo: one step equals U V^T
- QK-Clip and the shared rotary key in multi-head latent attention
- The whole training step in PyTorch

## Chapters
0:00 From gradient to update
1:29 The size of the step
4:40 Momentum
7:09 RMSProp
9:46 Adam and AdamW
13:13 The guards
16:01 Precision
18:54 Schedules
21:23 Batch size
23:52 Matrix updates
25:53 Muon
30:00 Shampoo
32:30 QK-Clip
35:08 The whole loop
36:20 Key references

## Key references
- Cohen et al. (2021), Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability, ICLR
- Tieleman & Hinton (2012), Lecture 6.5: RMSProp, Coursera
- Kingma & Ba (2014), Adam: A Method for Stochastic Optimization, ICLR 2015
- Loshchilov & Hutter (2017), Decoupled Weight Decay Regularization, ICLR 2019
- Goyal et al. (2017), Accurate, Large Minibatch SGD, arXiv:1706.02677
- Pascanu, Mikolov & Bengio (2013), On the difficulty of training recurrent neural networks, ICML
- Micikevicius et al. (2017), Mixed Precision Training, ICLR 2018
- Loshchilov & Hutter (2016), SGDR, ICLR 2017
- Hu et al. (2024), MiniCPM, arXiv
- McCandlish et al. (2018), An Empirical Model of Large-Batch Training, arXiv
- Jordan et al. (2024), Muon: An optimizer for hidden layers in neural networks
- Gupta, Koren & Singer (2018), Shampoo, ICML
- Kimi Team (2025), Kimi K2: Open Agentic Intelligence, arXiv:2507.20534

## Code

https://github.com/flavianv/ai-notes-code-public/tree/main/training-dynamics/films/f2b  (`lab.py` computes every number, `results.json` holds them, `make_narration.py` checks every spoken number against them; run `sh training-dynamics/run_checks.sh`).

Series: AI Notes · Transformer Training Dynamics (1 trainable at step zero · 2A gradients, module by module · 2B optimization tricks · 3 thousands of steps).
