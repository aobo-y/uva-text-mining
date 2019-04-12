import os
import json
import re
import pickle
import numbers
import random
import numpy as np
from math import log, sqrt
from collections import Counter

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.snowball import EnglishStemmer


STOP_WORD_PATH = './stop_words.txt'

tokenizer = TreebankWordTokenizer()
stemmer = EnglishStemmer()

stop_words = None

def normalize(token):
  token = re.sub(r'(^\W+|\W+$)', '', token)
  token = token.lower()
  if re.fullmatch(r'\d+(,\d{3})*(\.\d+)?', token):
    token = 'NUM'

  return stemmer.stem(token.strip())


with open(STOP_WORD_PATH) as sw_file:
  stop_words = sw_file.read().split('\n')
  stop_words = {normalize(w) for w in stop_words if w != ''}

class Review:
  def __init__(self, obj):
    # self.author = obj["Author"]
    self.rid = obj['ReviewID']
    self.date = obj['Date']
    self.label = 0 if float(obj['Overall']) < 4 else 1
    self.content = obj['Content']

    tokens = [normalize(token) for token in tokenizer.tokenize(self.content)]
    tokens = [token for token in tokens if token not in stop_words and token != '']
    self.tokens = Counter(tokens)

  def filter_vocab(self, vocab):
    self.features = Counter({k: v for k, v in self.tokens.items() if k in vocab})

  def set_vector(self, vocab, idf):
    tf = lambda k: 1 + log(self.tokens[k]) if k in self.tokens else 0

    self.vector = np.array([
      tf(k) * idf[k]
      for k in vocab
    ])


def read_folder(path):
  reviews = []

  for fname in os.listdir(path):
    fpath = os.path.join(path, fname)

    if os.path.isfile(fpath):
      with open(fpath) as f:
        obj = json.load(f)
        reviews += [Review(review) for review in obj["Reviews"]]
    else:
      reviews += read_folder(fpath)

  print('read', len(reviews), 'reviews from', path)

  return reviews
