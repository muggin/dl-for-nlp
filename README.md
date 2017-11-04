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

[GloVe](word-vectors.ipynb) based on ["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/pubs/glove.pdf) by Pennington et al.  
[word2vec](word-vectors.ipynb) based on ["Distributed Representations of Words and Phrases and their Compositionality"](https://arxiv.org/abs/1310.4546) by Mikolov et al.

#### Example Results
<div>
  <p>GloVe 100-dim embeddings projected to 2D using PCA</p>
  <img align="center" src="/misc/g-100-sem-relations.png" width=410>
  <img align="center" src="/misc/g-100-synt-verbs.png" width=410>
  
  <p>Word2Vec 100-dim embeddings projected to 2D using PCA</p>
  <img align="center" src="/misc/w2v-100-sem-relations.png" width=410>
  <img align="center" src="/misc/w2v-100-synt-adjectives.png" width=410>
</div>

## Neural Machine Translation

#### References
Implementation of a sequence-to-sequence network for Neural Machine Translation.

[Sequence-to-Sequence](seq-to-seq.ipynb) based on:
- ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) by Sustskever et al.
- ["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078) by Cho et al.

#### Example Results
<div>
  <p>Embeddings of words and phrases created with the Encoder network</p>
  <img align="center" src="/misc/mt-basic-word-emb.png" width=425>
  <img align="center" src="/misc/mt-basic-sent-emb.png" width=425>
</div>

<div>
  <p>Translations made with the trained model</p>
  <table>
  <tbody>
    <tr>
      <th>Source</th>
      <th>Translation</th>
    </tr>
    <tr>
      <td>the debate is closed</td>
      <td>die aussprache ist geschlossen</td>
    </tr>
    <tr>
      <td>this resolution has just been rejected</td>
      <td>diese entschließung wurde abgelehnt</td>
    </tr>
    <tr>
      <td>mr president the liberal group welcomes the conclusion of this agreement with jordan</td>
      <td>herr präsident die liberale fraktion begrüßt diesen abkommen</td>
    </tr>
  </tbody>
  </table>
</div>

## Attention-based Neural Machine Translation

#### References
Implementation of an attention-based sequence-to-sequence network for Neural Machine Translation.

[Attention-based SeqToSeq](attn-seq-to-seq.ipynb) based on:
- [Neural Machine Translation by Jointly Learning To Align and Translate](https://arxiv.org/abs/1409.0473) by Bahdanau et al.
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) by Luong et al.

#### Example Results
<div>
<p>Learned alignments between source and target sequences</p>
<img align="center" src="/misc/mt-attn-alignment-1.png" width=425>
<img align="center" src="/misc/mt-attn-alignment-2.png" width=425>
</div>
</br>

<div>
  <p>Translations made with the trained model</p>
  <table>
  <tbody>
    <tr>
      <th>Source</th>
      <th>Translation</th>
    </tr>
    <tr>
      <td>the debate is closed</td>
      <td>die aussprache ist geschlossen</td>
    </tr>
    <tr>
      <td>this resolution has just been rejected</td>
      <td>diese entschließung wurde nur abgelehnt</td>
    </tr>
    <tr>
      <td>mr president the liberal group welcomes the conclusion of this agreement with jordan</td>
      <td>herr präsident die fraktion begrüßt die schlussfolgerung dieses abkommens mit jordanien</td>
    </tr>
  </tbody>
  </table>
</div>
