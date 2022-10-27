from __future__ import print_function
import sys
from setuptools import setup

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

version = '1.0'

setup(name='agml-bias-score',
      version=version,
      description='Implementation and wrapper of bias scores for NLP',
      url='https://github.com/UBI-AGML-NLP/EmbeddingBiasScores',
      packages=['geometrical_bias', 'lipstick_bias'],
      install_requires=INSTALL_REQUIRES,
      )
