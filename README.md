# Deep Learning for NLP
The goal of this project was to explore different applications of (Deep) Neural Networks to the field of Natural Language Processing. The workload consisted of a literature study of the field and an implementational part. The final report will be made available.

This project was carried out as part of the DD2465 "Advanced, Individual Course in Computer Science" course at [KTH Royal Institute of Technology](http://kth.se).

Content:
- [Word Embeddings](#word-embeddings)
- [Neural Machine Translation](#neural-machine-translation)
- [Attention-based Neural Machine Translation](#attention-based-neural-machine-translation)

## Word Embeddings
> "You shall know a word by the company it keeps." - Firth

#### References
Implementation of two algorithms for building distributed word representations.  

GloVe based on ["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/pubs/glove.pdf) by Pennington et al.  
word2vec based on ["Distributed Representations of Words and Phrases and their Compositionality"](https://arxiv.org/abs/1310.4546) by Mikolov et al.

#### Example Results
<div>
  GloVe 100-dim embeddings projected to 2D using PCA
  <img align="center" src="/misc/g-100-sem-relations.png" width=415>
  <img align="center" src="/misc/g-100-synt-verbs.png" width=415>
  Word2Vec 100-dim embeddings projected to 2D using PCA
  <img align="center" src="/misc/w2v-100-sem-relations.png" width=415>
  <img align="center" src="/misc/w2v-100-synt-adjectives.png" width=415>
</div>

## Neural Machine Translation

#### References
Implementation of a sequence-to-sequence network for Neural Machine Translation.

SeqToSeq based on:
- ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) by Sustskever et al.
- ["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078) by Cho et al.

#### Example Results
<div>
  Embeddings of words and phrases created using the Encoder network
  <img align="center" src="/misc/mt-basic-word-emb.png" width=425>
  <img align="center" src="/misc/mt-basic-sent-emb.png" width=425>
</div>


## Attention-based Neural Machine Translation
Implementation of an attention-based sequence-to-sequence network for Neural Machine Translation.
