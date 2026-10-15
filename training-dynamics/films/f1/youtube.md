# Transformer Training Dynamics 1: Trainable at step zero

## Description

Before the first training step, signals and gradients have to travel through every layer of a Transformer at a sensible size. We follow a courier of size one through 48 layers and derive, with numbers you can rerun, the seven design choices that keep both the forward signal and the gradient alive.

Every number in the film is computed or measured by a script you can rerun (NumPy, CPU), and each spoken number is asserted against the script before the voice-over is generated.

What is derived or checked:
- Variance, and why a mean of zero is not a size
- Why sums of random numbers grow like the square root of d
- A random matrix: weight variance 1/d (He: 2/d for ReLU), forward against backward scale
- The 1/sqrt(d_k) in attention, derived and measured
- What training changes: alignment, softmax shifts and sharpening
- Softmax saturation and the cross-entropy gradient p - y
- Depth: Jacobian products, residual branches and the 1/sqrt(L) scaling
- LayerNorm, RMSNorm and where the norm sits (a gradient measurement on 24 blocks)
- The MLP as a key-value memory; its gradients; dead ReLU units
- Activation functions from sigmoid to SwiGLU and the 8d/3 width
- QK-norm and soft-capping against runaway attention scores

## Chapters
0:00 The courier
1:36 The route
3:01 Mean is not magnitude
4:26 Why square roots appear
6:20 What a random matrix does
8:33 Forward and backward scale
10:38 The attention scaling
12:41 What training changes
14:20 The jacobian, and softmax saturation
18:31 Keeping scores in range
20:21 Depth and residual connections
22:21 Where normalization acts
24:37 The MLP as a key-value memory
27:33 Activation functions
31:24 Derive it yourself
32:38 Key references

## Key references
- Glorot & Bengio (2010), Understanding the difficulty of training deep feedforward neural networks, AISTATS
- He, Zhang, Ren & Sun (2015), Delving Deep into Rectifiers, ICCV
- Vaswani et al. (2017), Attention Is All You Need, NeurIPS
- Ba, Kiros & Hinton (2016), Layer Normalization, arXiv:1607.06450
- Zhang & Sennrich (2019), Root Mean Square Layer Normalization, NeurIPS
- Xiong et al. (2020), On Layer Normalization in the Transformer Architecture, ICML
- Geva et al. (2021), Transformer Feed-Forward Layers Are Key-Value Memories, EMNLP
- Meng et al. (2022), Locating and Editing Factual Associations in GPT, NeurIPS
- Hendrycks & Gimpel (2016), Gaussian Error Linear Units, arXiv:1606.08415
- Shazeer (2020), GLU Variants Improve Transformer, arXiv:2002.05202
- Dehghani et al. (2023), Scaling Vision Transformers to 22 Billion Parameters, ICML
- Gemma Team (2024), Gemma 2: Improving Open Language Models at a Practical Size

## Code

`transformer-training-dynamics/v2/f1/lab.py` (results in `results.json`), `make_narration.py` (asserts every spoken number), `scenes*.part.js` (the film), in github.com/flavianv/ai_tutorials.

Series: AI Notes · Transformer Training Dynamics (1 trainable at step zero · 2A gradients, module by module · 2B optimization tricks · 3 thousands of steps).
