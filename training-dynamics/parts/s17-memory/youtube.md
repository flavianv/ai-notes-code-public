# Attention as memory: from softmax to the delta rule

## Description

Attention is a memory: each token writes a key and a value, each query reads. We follow one memory through four designs with the same test (write n random pairs, read every key back): softmax keeps everything exactly and grows, linear attention adds into a fixed matrix and blends, the delta rule writes only the error, and gates let a fixed memory forget. Hybrids keep a few exact layers.

Every number in the film is computed or measured by a script you can rerun, and each spoken number is asserted against it before the voice-over is generated.

## Chapters
0:00 Attention as a memory
0:53 Softmax: keep everything
2:34 Linear attention: add everything
4:29 The delta rule: write the error
6:44 Gates and hybrids
8:20 In code, and what to remember
9:17 Key references

## Key references
- Katharopoulos et al. (2020), Transformers are RNNs, ICML
- Schlag, Irie & Schmidhuber (2021), Linear Transformers Are Secretly Fast Weight Programmers, ICML
- Yang et al. (2024), Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv:2406.06484
- Yang, Kautz & Hatamizadeh (2024), Gated Delta Networks, arXiv:2412.06464
- Kimi Team (2025), Kimi Linear, arXiv:2510.26692

## Code

In github.com/flavianv/ai_tutorials, folder transformer-training-dynamics/v2/.

Series: AI Notes · How Transformers Train (standalone parts).
