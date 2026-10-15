# Induction heads, layer by layer: how a transformer learns to copy a pattern

## Description

Seen A, B once; shown A again; predict B. We find the two-head circuit that does it in a tiny attention-only model (previous-token heads in layer one, induction heads in layer two), watch the loss plateau and then drop as the circuit forms, and switch heads off one at a time and in groups to see which parts matter. Results of three seeds from the Understanding Transformers A1 experiment.

Every number in the film is computed or measured by a script you can rerun, and each spoken number is asserted against it before the voice-over is generated.

## Chapters
0:00 The question
0:45 The idea: two steps
2:24 The experiment
3:34 What each layer learns
5:15 Switching parts off
6:19 Limits, PyTorch and what to remember
7:34 Key references

## Key references
- Elhage et al. (2021), A Mathematical Framework for Transformer Circuits
- Olsson et al. (2022), In-context Learning and Induction Heads

## Code

In github.com/flavianv/ai_tutorials, folder transformer-training-dynamics/v2/.

Series: AI Notes · How Transformers Train (standalone parts).
