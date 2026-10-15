# Transformer Training Dynamics 3: Thousands of steps

## Description

Run a small model for thousands of steps and watch what it builds: an induction circuit that forms while the loss barely moves, a network that memorizes and then generalizes, and the reason the simple solution wins. Then linear attention, delta attention, the 2026 frontier models and LoRA-style adaptation, each with numbers you can rerun.

Every number in the film is computed or measured by a script you can rerun (NumPy, CPU), and each spoken number is asserted against the script before the voice-over is generated.

What is derived or checked:
- Matching and message maps; a change of basis that leaves every score unchanged
- Gradients as sums of outer products; effective rank; superposition
- The induction circuit: measured scores and ablations
- Loss curves that hide the formation of a circuit
- Fitting random labels; grokking with and without weight decay
- Implicit bias: gradient descent finds the minimum-norm solution; double descent
- Width, mu-P and scaling laws
- Linear attention as a recurrence
- Delta attention: the delta rule, the chunked form, Gated DeltaNet and Kimi Delta Attention
- What the 2026 frontier models change
- LoRA, QLoRA, VeRA and DoRA: initialization, rank, 4-bit levels
- Making derivations falsifiable: the checks of all three films

## Chapters
0:00 Thousands of steps
0:56 Matching and message content
2:26 How useful structure accumulates
4:10 A circuit learns a reusable lookup
5:48 Loss curves can hide internal change
7:20 Fitting noise
8:37 Grokking: memorize first, generalize later
10:16 Why the rule wins
11:51 Width, mu-P and scaling laws
13:44 Linear attention
15:29 Delta attention
18:53 What the 2026 frontier models change
20:34 Adaptation dynamics: LoRA and friends
23:00 Make the derivations falsifiable
24:12 Derive it yourself
25:06 Key references

## Key references
- Elhage et al. (2021), A Mathematical Framework for Transformer Circuits
- Elhage et al. (2022), Toy Models of Superposition
- Olsson et al. (2022), In-context Learning and Induction Heads
- Zhang et al. (2017), Understanding deep learning requires rethinking generalization, ICLR
- Power et al. (2022), Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
- Nanda et al. (2023), Progress measures for grokking via mechanistic interpretability, ICLR
- Keskar et al. (2017), On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima, ICLR
- Belkin et al. (2019), Reconciling modern machine-learning practice and the classical bias-variance trade-off, PNAS
- Nakkiran et al. (2019), Deep Double Descent, ICLR 2020
- Kaplan et al. (2020), Scaling Laws for Neural Language Models
- Hoffmann et al. (2022), Training Compute-Optimal Large Language Models
- Yang et al. (2022), Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer
- Katharopoulos et al. (2020), Transformers are RNNs, ICML
- Schlag, Irie & Schmidhuber (2021), Linear Transformers Are Secretly Fast Weight Programmers, ICML
- Yang et al. (2024), Parallelizing Linear Transformers with the Delta Rule over Sequence Length, NeurIPS
- Yang, Kautz & Hatamizadeh (2024), Gated Delta Networks, ICLR 2025
- Kimi Team (2025), Kimi Linear: An Expressive, Efficient Attention Architecture, arXiv:2510.26692
- Hu et al. (2021), LoRA, arXiv:2106.09685
- Hayou, Ghosh & Yu (2024), LoRA+, arXiv:2402.12354
- Dettmers et al. (2023), QLoRA, arXiv:2305.14314
- Kopiczko et al. (2023), VeRA, arXiv:2310.11454
- Liu et al. (2024), DoRA, arXiv:2402.09353

## Code

`transformer-training-dynamics/v2/f3/lab.py` (results in `results.json`), `make_narration.py` (asserts every spoken number), `scenes*.part.js` (the film), in github.com/flavianv/ai_tutorials.

Series: AI Notes · Transformer Training Dynamics (1 trainable at step zero · 2A gradients, module by module · 2B optimization tricks · 3 thousands of steps).
