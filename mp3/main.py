import os
import json
import re
import pickle
import numbers
import random
import math
from collections import Counter
from decimal import Decimal
from statistics import mean, stdev

from data import read_folder

D = Decimal

SAVE_PATH = './save.pickle'



def count_grams(reviews):
  uni_counts = {}
  bi_counts = {}

  for review in reviews:
    prev_token = None

    for token in review.tokens:
      if token in uni_counts:
        uni_counts[token] += 1
      else:
        uni_counts[token] = 1

      if prev_token in bi_counts:
        if token in bi_counts[prev_token]:
          bi_counts[prev_token][token] += 1
        else:
          bi_counts[prev_token][token] = 1
      elif prev_token != None:
        bi_counts[prev_token] = {token: 1}

      prev_token = token

  return uni_counts, bi_counts



def add_vocab(vocab, test_reviews):
  v = set(vocab)
  for review in test_reviews:
    for t in review.tokens:
      v.add(t)
  return list(v)

def information_gain(reviews, words):
  # probs of positive / negative review
  l_counts = Counter([review.label for review in reviews])
  l_probs = {l: c / len(reviews) for l, c in l_counts.items()}

  print('label probs')
  print(l_probs)

  entropy = lambda probs: sum([
    0 if p == 0 else p * math.log(p) for k, p in l_probs.items()
  ])

  words_ig = {}
  for word in words:
    # probs of review with / without the word
    w_counts = Counter([
      word in review.tokens for review in reviews
    ])

    w_probs = {k: v / sum(w_counts.values()) for k, v in w_counts.items()}

    # probs of positive / negative review with the word
    l_w_counts = Counter([
      review.label for review in reviews if word in review.tokens
    ])

    l_w_probs = {k: v / sum(l_w_counts.values()) for k, v in l_w_counts.items()}

    # probs of positive / negative review without the word
    l_n_w_counts = Counter([
      review.label for review in reviews if word not in review.tokens
    ])

    l_n_w_probs = {k: v / sum(l_n_w_counts.values()) for k, v in l_n_w_counts.items()}

    words_ig[word] = - entropy(l_probs) \
      + w_probs[True] * entropy(l_w_probs) \
      + w_probs[False] * entropy(l_n_w_probs)

  return words_ig

def document_frequency(reviews):
  df = Counter()
  for review in reviews:
    df.update(review.tokens.keys())

  return df

def select_features(reviews):
  df = document_frequency(reviews)
  print('Vocab size:', len(df))

  vocab = [w for w, v in df.items() if v >= 10]
  print('Vocab size (df >= 10):', len(vocab))

  igs = information_gain(reviews, vocab)




def main():
  if os.path.isfile(SAVE_PATH):
    print('load save', SAVE_PATH)

    with open(SAVE_PATH,'rb') as pf:
      reviews = pickle.load(pf)
  else:
    reviews = read_folder('./yelp/train')
    # test_reviews = read_folder('./yelp/test')
    with open(SAVE_PATH, 'wb') as pf:
      pickle.dump(reviews, pf)

  select_features(reviews)



main()

