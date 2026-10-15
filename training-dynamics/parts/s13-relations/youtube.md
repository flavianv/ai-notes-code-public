# Relations and outer-product memory: where does "king minus man plus woman" live?

## Description

Where does a model keep a fact, and where a relation? A matrix stores facts as a sum of outer products (with exact crosstalk sqrt((n-1)/d)); relations are directions, so king - man + woman = queen exactly in a toy embedding (36 of 36 analogies); and a network can apply a relation to a pair it has never seen only if its representation carries the attributes (3.9% vs 99.7% held-out accuracy), shown layer by layer with linear probes.

Every number in the film is computed or measured by a script you can rerun, and each spoken number is asserted against it before the voice-over is generated.

## Chapters
0:00 The question
0:37 Memory as a sum of outer products
2:57 Relations are directions
5:02 Relations, layer by layer
7:09 In code, limits and what to remember
8:17 Key references

## Key references
- Kohonen (1972), Correlation matrix memories, IEEE Trans. Computers
- Mikolov, Yih & Zweig (2013), Linguistic Regularities in Continuous Space Word Representations, NAACL
- Geva et al. (2021), Transformer Feed-Forward Layers Are Key-Value Memories, EMNLP
- Meng et al. (2022), Locating and Editing Factual Associations in GPT, NeurIPS
- Elhage et al. (2022), Toy Models of Superposition

## Code

In github.com/flavianv/ai_tutorials, folder transformer-training-dynamics/v2/.

Series: AI Notes · How Transformers Train (standalone parts).
