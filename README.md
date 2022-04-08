# EmbeddingBiasScores
Implementations and wrapper of bias scores for text embeddings.


## Requirements
All requirements are listed in ```requirements.txt```.  
In the notebooks, we use a wrapper for huggingface models. You can install the lib from source:  
https://github.com/UBI-AGML-NLP/Embeddings

## Installation
Download this repository and install via ```pip install .```

## Example
A minimalistic example for the usage of implemented bias scores can be found in ```example.ipynb```.

## Implemented Scores
So far, we implemented geometrical bias scores that measure the association of embeddings with protected groups. Other scores and bias test will follow.  
Papers that introduced the scores are linked.

#### Geometrical Bias scores

- [SAME](https://arxiv.org/abs/2111.07864)
- [WEAT](https://www.science.org/doi/abs/10.1126/science.aal4230)
- [generalized WEAT](https://dl.acm.org/doi/pdf/10.1145/3306618.3314270)
- [Direct Bias](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)
- [RIPA](https://arxiv.org/abs/1908.06361)
- [MAC](https://arxiv.org/abs/1904.04047)
