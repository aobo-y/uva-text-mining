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

  entropy = lambda probs: sum([
    0 if p == 0 else p * math.log(p) for k, p in probs.items()
  ])

  words_ig = {}
  for word in words:
    w_reviews = [review for review in reviews if word in review.tokens]

    # probs of review with / without the word
    w_counts = {True: len(w_reviews), False: len(reviews) - len(w_reviews)}

    w_probs = {k: v / sum(w_counts.values()) for k, v in w_counts.items()}

    # probs of positive / negative review with the word
    l_w_counts = Counter([
      review.label for review in w_reviews
    ])

    l_w_probs = {k: v / sum(l_w_counts.values()) for k, v in l_w_counts.items()}

    # probs of positive / negative review without the word
    l_n_w_counts = {k: v - l_w_counts[k] for k, v in l_counts.items()}

    l_n_w_probs = {k: v / sum(l_n_w_counts.values()) for k, v in l_n_w_counts.items()}

    words_ig[word] = - entropy(l_probs) \
      + w_probs[True] * entropy(l_w_probs) \
      + w_probs[False] * entropy(l_n_w_probs)

  return words_ig

def chi_square(reviews, words):
  words_cs = {}
  pos_reviews = [review for review in reviews if review.label == 1]
  neg_reviews = [review for review in reviews if review.label == 0]

  for word in words:
    pos_counts = Counter(word in review.tokens for review in pos_reviews)
    neg_counts = Counter(word in review.tokens for review in neg_reviews)

    A = pos_counts[True]
    B = pos_counts[False]
    C = neg_counts[True]
    D = neg_counts[False]

    words_cs[word] = (A + B + C + D) * ((A * D - B * C) ** 2) \
      / ((A + C) * (B + D) * (A + B) * (C + D))

  return words_cs


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

  css = chi_square(reviews, vocab)

  css = {w: v for w, v in css.items() if v >= 3.841}
  print('Vocab size (chi square >= 3.841):', len(css))

  igs = information_gain(reviews, vocab)

  css = sorted(css.items(), key=lambda i: i[1], reverse=True)
  igs = sorted(igs.items(), key=lambda i: i[1], reverse=True)

  if len(css) >= 5000:
    css = css[:5000]
  if len(igs) >= 5000:
    igs = igs[:5000]

  print('Top 20 Information Gain')
  print(igs[:20])
  print('Top 20 Chi Square')
  print(css[:20])

  vocab = set(i[0] for i in css).union(i[0] for i in igs)

  print('Final Vocab size:', len(vocab))

  return vocab




def main():
  if os.path.isfile(SAVE_PATH):
    print('load save', SAVE_PATH)

    with open(SAVE_PATH,'rb') as pf:
      corpus, vocab = pickle.load(pf)
  else:
    corpus = read_folder('./yelp')
    vocab = select_features(corpus)

    print('Corpus size:', len(corpus))
    for d in corpus:
      d.filter_vocab(vocab)
    corpus = [d for d in corpus if len(d.features) > 5]
    print('Corpus size (feature > 5):', len(corpus))

    with open(SAVE_PATH, 'wb') as pf:
      pickle.dump((corpus, vocab), pf)


main()

