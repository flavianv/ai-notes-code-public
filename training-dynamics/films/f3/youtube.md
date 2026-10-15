# Transformer Training Dynamics 3: What training builds, and what scale changes

## Description

Run a model for thousands of steps and ask what training builds, and what changes when the model gets bigger. Seven stops, each with the intuition, the math with the numbers plugged in, a plot and the code: the induction circuit layer by layer, facts and relations in an outer-product memory, attention as a memory from softmax to the delta rule, scaling laws, mu-P, grokking, and the newest frontier model read against all of it.

Every number in the film is computed or measured by a script you can rerun (NumPy, CPU), and each spoken number is asserted against the script before the voice-over is generated.

What is derived or checked:
- The induction circuit: a model learns to copy a pattern with two heads in two layers, measured layer by layer with ablations
- An outer-product memory stores facts with exact crosstalk; relations are directions (king - man + woman = queen); a network reuses a relation only if its representation carries the attributes
- Attention as a memory: softmax keeps every pair, linear attention adds, the delta rule writes the error, gates forget; hybrids keep a few exact layers
- Scaling laws: two gaps above a floor, the compute-optimal split, and why the published 20 tokens per parameter is a fit
- mu-P: why one step changes the output by eta times the width, and the recipe that makes the learning rate transfer
- Grokking and why the rule wins: memorize first, generalize later, minimum-norm solutions, double descent
- The 2026 frontier models read against the series (Kimi K3, a dated snapshot)

## Chapters
0:00 Thousands of steps: what training builds, and what scale changes
0:57 Induction heads, layer by layer
1:42 The idea: two steps
3:22 The experiment
4:32 What each layer learns
6:13 Switching parts off
7:17 Limits, PyTorch and what to remember
8:31 Relations and outer-product memory
9:08 Memory as a sum of outer products
11:30 Relations are directions
13:35 Relations, layer by layer
15:42 In code, limits and what to remember
16:51 Attention as memory: softmax to the delta rule
17:44 Softmax: keep everything
19:23 Linear attention: add everything
21:18 The delta rule: write the error
23:33 Gates and hybrids
25:10 In code, and what to remember
26:07 Scaling laws: bigger model or more data?
26:54 A floor and two gaps
29:21 Splitting the budget
31:24 The published rule
33:27 Limits, and the code
34:52 mu-P: why the learning rate stops transferring
35:32 One step on a wide layer
37:28 The mu-P recipe
39:31 Does the learning rate transfer?
41:06 In code, limits and what to remember
42:14 Grokking: why the rule wins
42:55 Fitting noise
44:12 Grokking
45:48 Why the rule wins
47:23 What to remember
48:05 What changed in the 2026 frontier models
48:53 Signals: scores and depth
50:17 The shape of the update
51:03 Scale: context and parameters
52:36 The answer sheet and limits
53:43 Key references

## Key references
- Elhage et al. (2021), A Mathematical Framework for Transformer Circuits
- Olsson et al. (2022), In-context Learning and Induction Heads
- Kohonen (1972), Correlation matrix memories, IEEE Trans. Computers
- Mikolov, Yih & Zweig (2013), Linguistic Regularities in Continuous Space Word Representations, NAACL
- Geva et al. (2021), Transformer Feed-Forward Layers Are Key-Value Memories, EMNLP
- Meng et al. (2022), Locating and Editing Factual Associations in GPT, NeurIPS
- Elhage et al. (2022), Toy Models of Superposition
- Katharopoulos et al. (2020), Transformers are RNNs, ICML
- Schlag, Irie & Schmidhuber (2021), Linear Transformers Are Secretly Fast Weight Programmers, ICML
- Yang et al. (2024), Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv:2406.06484
- Yang, Kautz & Hatamizadeh (2024), Gated Delta Networks, arXiv:2412.06464
- Kimi Team (2025), Kimi Linear, arXiv:2510.26692
- Kaplan et al. (2020), Scaling Laws for Neural Language Models, arXiv:2001.08361
- Hoffmann et al. (2022), Training Compute-Optimal Large Language Models, arXiv:2203.15556
- Besiroglu et al. (2024), Chinchilla Scaling: A replication attempt, Epoch AI
- Sardana & Frankle (2023), Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws, arXiv:2401.00448
- Yang & Hu (2021), Feature Learning in Infinite-Width Neural Networks, ICML
- Yang et al. (2022), Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer, arXiv:2203.03466
- Zhang et al. (2017), Understanding deep learning requires rethinking generalization, ICLR
- Power et al. (2022), Grokking, arXiv
- Nanda et al. (2023), Progress measures for grokking via mechanistic interpretability, ICLR
- Belkin et al. (2019), Reconciling modern machine-learning practice and the classical bias-variance trade-off, PNAS
- Kimi Team (2026), Kimi K3: Open Frontier Intelligence, arXiv:2607.24653
- Kimi Team (2025), Kimi K2: Open Agentic Intelligence, arXiv:2507.20534
- Kimi Team (2025), Kimi Linear: An Expressive, Efficient Attention Architecture, arXiv:2510.26692

## Code

https://github.com/flavianv/ai-notes-code-public/tree/main/training-dynamics/films/f3  (`lab.py` and the part folders under `parts/` compute every number, `results.json` holds them, `make_narration.py` checks every spoken number against them; run `sh training-dynamics/run_checks.sh`).

Series: AI Notes · Transformer Training Dynamics (1 trainable at step zero · 2A gradients, module by module · 2B optimization tricks · 3 what training builds, and what scale changes).
