# Transformer Training Dynamics 2A: Gradients, module by module

## Description

Every weight gets a share of the loss. We follow the penalty backward through each module of a Transformer, with the numbers plugged in at every step: the loss, one linear layer, one matrix shared by many tokens, the embeddings, residual connections, norm layers, the MLP and attention, and why a route that already helps gains more attention.

Every number in the film is computed or measured by a script you can rerun (NumPy, CPU), and each spoken number is asserted against the script before the voice-over is generated.

What is derived or checked:
- The loss gradient p - e_y on a four-word example
- The chain rule through one linear layer: G_W = X^T G_Y, G_X = G_Y W^T, with the penalty fanning out to every input and weight
- One matrix, many tokens: the update is a sum over the words
- Sparse gradients of the embedding table
- The residual connection: a copy of the penalty along the straight path
- RMS norm backward: the gradient is perpendicular to the input
- The MLP as key-value memory, backward
- Attention backward: G_V, G_A, G_S, G_Q, G_K, checked against finite differences
- Why a route gains attention

## Chapters
0:00 The penalty
1:25 The loss
4:36 One linear layer
8:13 One matrix, many tokens
11:26 The embeddings
13:52 The residual connections
16:21 The norm layers
19:05 The MLP
22:23 Attention
25:35 Routes
28:13 One block, who gets what
29:58 Key references

## Key references
- Rumelhart, Hinton & Williams (1986), Learning representations by back-propagating errors, Nature
- Baydin et al. (2018), Automatic differentiation in machine learning: a survey, JMLR
- He, Zhang, Ren & Sun (2016), Deep Residual Learning for Image Recognition, CVPR
- Ba, Kiros & Hinton (2016), Layer Normalization, arXiv:1607.06450

## Code

`transformer-training-dynamics/v2/f2a/lab.py` (results in `results.json`), `make_narration.py` (asserts every spoken number), `scenes*.part.js` (the film), in github.com/flavianv/ai_tutorials.

Series: AI Notes · Transformer Training Dynamics (1 trainable at step zero · 2A gradients, module by module · 2B optimization tricks · 3 thousands of steps).
