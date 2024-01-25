# NPL-A1
NLP assignment from Asian Institute of Technology

# Table of Contents
1. [ID](#ID)
2. [Test1](#Test1)
3. [Test2](#Test2)
4. [Test3](#Test3)

## ID
Kaung Htet Cho (st124092)

## Test1

corpus - NLTK reuters

total word - 75698

vocab - 8557

epoch - 1000

embed_size - 30

batch_size - 2

window_size - 2

I used same hypermeters for three models, total word is limited to 75698 due to the computational limitation of GloVe's weight dictionary.

## Test2

| Model          | Window Size | Training Loss | Training time | Syntactic Accuracy | Semantic accuracy |
|----------------|-------------|---------------|---------------|--------------------|-------------------|
| Skipgram       |    2         |      35.739899   |        04m 43s       |            0        |        0          |
| Skipgram (NEG) |      2       |       04m 34s        |         04m 34s      |        0            |        0          |
| Glove          |          2   |   2.528146       |         19s      |              0      |        0          |
| Glove (Gensim) | -           | -             | -             |        55.45           |        93.87        |

For syntactic and semantic accuracy calculations, I used offset word technique in word2vec paper and then calculate the cosine-similartiy and find the accuracy.

## Test3

For skipgram, skipgramNEG and GloVe models, implemented cosine similarity between the user input words, vectorized the input words and then results 10 most similar words.
For gensim, I used its built-in similarity calculation and print out 10 most similar words 