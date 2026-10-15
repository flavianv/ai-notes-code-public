# Scaling laws: a bigger model or more data? (and why "20 tokens per parameter" is a fit, not a law)

## Description

You have a fixed compute budget. We derive the compute-optimal split from the two-gap form L = E + A/N^alpha + B/D^beta under C = 6ND, test it on 24 small training runs, and then plug in the published Chinchilla constants: taken literally they give about 93 tokens per parameter at Chinchilla's budget, while Epoch AI's refit gives 18.4, close to the 20 that was used.

Every number in the film is computed or measured by a script you can rerun, and each spoken number is asserted against it before the voice-over is generated.

## Chapters
0:00 The question
0:46 A floor and two gaps
3:13 Splitting the budget
5:16 The published rule
7:19 Limits, and the code
8:44 Key references

## Key references
- Kaplan et al. (2020), Scaling Laws for Neural Language Models, arXiv:2001.08361
- Hoffmann et al. (2022), Training Compute-Optimal Large Language Models, arXiv:2203.15556
- Besiroglu et al. (2024), Chinchilla Scaling: A replication attempt, Epoch AI
- Sardana & Frankle (2023), Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws, arXiv:2401.00448

## Code

In github.com/flavianv/ai_tutorials, folder transformer-training-dynamics/v2/.

Series: AI Notes · How Transformers Train (standalone parts).
