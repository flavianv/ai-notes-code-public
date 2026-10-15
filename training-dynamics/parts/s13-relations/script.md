# Relations and outer-product memory: where does a model keep "king minus man plus woman"?

Standalone part 13. Three small experiments in `lab.py` (pure NumPy); numbers in `results.json`.

---

## 1. the question

**Voice-over:**
Where does a model keep a fact, like France and Paris? And where does it keep a relation, like the difference between man and woman, so that king minus man plus woman lands on queen? This film answers both with small experiments, and ends by asking what happens to a relation layer by layer.

You need three things. A dot product measures how much two vectors point the same way. The outer product of a key and a value is a matrix that sends the key to the value. And an embedding is a vector that stands for a word or an entity.

**Reader:** "Facts in matrices. Relations in directions."

---

## 2. memory as a sum of outer products

**Voice-over:**
Here is the intuition. To store a fact, write the pair into a matrix: the key says what to look for, and the value says what to return. Store several pairs by adding their matrices. To read, multiply the matrix by a key. The second matrix of an MLP is exactly this: a sum, over hidden units, of a value times a key.

Now we name the pieces. The memory is W, the sum over stored pairs of the value times the transposed key. Reading key j gives W times k j. Expanding the sum, that is v j times the squared length of k j, plus the sum of every other value times its key's dot product with k j. The first term is the fact. The second is crosstalk.

Now the math, with the numbers plugged in. Store two pairs. Key one is 1, 0, 0, with value 1, 0. Key two is 0.6, 0.8, 0, with value 0, 1. The two keys have a dot product of 0.6. Reading key one returns 1, 0.6: the right value, plus 0.6 of the other, exactly the dot product. With orthogonal keys the crosstalk is zero, and in a width of 64 we measure an error below 10 to the minus 15 for up to 64 stored pairs.

Random keys are not orthogonal. With unit keys and unit values in a width d, the squared error is about n minus 1 over d, so the relative error is the square root of n minus 1 over d. At a width of 64, that is 0.48 for 16 pairs, 0.99 for 64 and 1.41 for 128. We measured 0.47, 0.99 and 1.41.

Here is what it looks like. The measured error follows the curve, and orthogonal keys stay at zero. A matrix can hold as many facts as its width, if the keys do not overlap. Beyond that, each new fact adds noise to all the others. This is why features are packed in superposition, and why the keys that a model learns try to stay apart.

**Reader:** "Value times key. Add them up."

---

## 3. relations are directions

**Voice-over:**
Here is the intuition. Facts need a memory. A relation like man to woman is different: it is a shift. If a model represents each entity as a sum of its attributes, then the change from man to woman is the same vector as the change from king to queen. Subtracting one entity from another cancels everything they share.

Now we name the pieces. Take eight entities: man, woman, boy, girl, king, queen, prince and princess. They are built from three binary attributes: gender, age and royalty. We train a small embedding, four numbers per entity, to predict the words that go with each attribute: he or she, works or plays, crown or market. Mikolov and colleagues showed in 2013 that learned word vectors support this kind of arithmetic.

Now the math, with the numbers plugged in. We fit each embedding as a constant, plus gender times a direction, plus age times a direction, plus royalty times a direction. The fit leaves 2 percent of the variance unexplained. The three directions have lengths 3.0, 3.3 and 3.4, and are almost orthogonal: their cosines are below 0.02.

So woman minus man, girl minus boy, queen minus king and princess minus prince are the same vector. Their pairwise cosines are 1.00, and their lengths are 3.0, 3.0, 3.1 and 3.0. Compute king minus man plus woman and compare it with the other entities: queen has a cosine of 1.00, and the next best, princess, 0.28. Over all 36 analogies, along each of the three attributes, the nearest entity is the right one 36 times.

Here is what it looks like. The eight embeddings sit at the corners of a cube, one edge direction per attribute. A relation is an edge, and every edge in the same direction is the same relation.

**Reader:** "A relation is an edge of the cube."

---

## 4. relations, layer by layer

**Voice-over:**
Here is the intuition. Knowing that a relation is a direction is not the same as being able to apply it. A network that applies "flip this attribute" to an entity it has not seen with that relation must reuse what it learned from other entities. That only works if the representation lets it.

Now we name the pieces. We build a larger world: 32 entities, each a combination of five binary attributes, and five relations, each flipping one attribute. That gives 160 pairs of an entity and a relation. We train a small network on 96 pairs and test it on the other 64, which it has never seen. The network has two hidden layers of 64 units. We try two kinds of input for the entity: a one-hot identifier, or the learned embedding from the last chapter.

Now the results, with the numbers. With one-hot identifiers, the network gets 100 percent of the training pairs right, and 3.9 percent of the held-out pairs, against 3.1 percent for random guessing. It has memorized the pairs. With learned embeddings, it gets 99.7 percent of the held-out pairs right.

Now we look at each layer. We train a linear probe to read each attribute from the representation, on 24 entities, and test it on the 8 that are left out. With one-hot identifiers, the input carries no structure: the probe gets 38 percent. The first hidden layer reaches 89 percent and the second 85 percent: the network builds attribute features layer by layer, but not enough to apply a relation to a new pair. With learned embeddings, the probe reads the attributes at 100 percent at the input and at both hidden layers, because the structure was there from the start.

Here is what it looks like. Two bars for held-out accuracy, and a line per input type for the probe accuracy at each layer. Structure in the representation is what lets a relation transfer.

**Reader:** "Learn the cube first. Then apply the edge."

---

## 5. in code, and what to remember

**Voice-over:**
And here is the same thing in code. The memory is a sum of outer products, the read is a matrix-vector product, and a relation is a difference of two vectors added to a third, followed by a nearest-neighbor search.

Some limits. These are toy worlds. The entities are exact combinations of attributes, and the embeddings come from a synthetic objective that makes the attributes add up. Real models learn relations that hold only approximately, many facts share the same directions in superposition, and the evidence that facts live in mid-layer MLPs comes from trained models, as reported by Geva and by Meng and their colleagues.

To remember: a matrix stores facts as a sum of outer products, and reading it returns the fact plus crosstalk from every other pair. A relation is a direction, and the same direction applies to every entity that shares the structure. A network can only reuse a relation on a new pair if its representation carries the attributes. Next, the film on attention as memory follows the same memory idea through the architectures of newer models.

**Reader:** "Facts add up. Relations are edges."

---

## 6. key references

**Voice-over:**
The key references are on screen.

**Reader:** "Read the papers. They are good."
