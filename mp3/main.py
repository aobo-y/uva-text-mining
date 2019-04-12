import os
import json
import re
import pickle
import numbers
import random
import math
import time
from collections import Counter
from statistics import mean, stdev
import matplotlib.pyplot as plt


from data import read_folder
from language_model import LanguageModel
from knn import KNN


SAVE_PATH = './save.pickle'


def build_lang_models(reviews, vocab, delta = 0.1):
  pos_counts = Counter()
  neg_counts = Counter()

  for review in reviews:
    if review.label:
      pos_counts.update(review.features.keys())
    else:
      neg_counts.update(review.features.keys())

  pos_model = LanguageModel(pos_counts, vocab, delta)
  neg_model = LanguageModel(neg_counts, vocab, delta)

  return pos_model, neg_model




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


def rank_nb_log_ratio(pos_model, neg_model):
  vocab = pos_model.vocab
  log_ratios = {
    word: math.log(pos_model.probs[word] / neg_model.probs[word])
    for word in vocab
  }

  print('bland:', pos_model.probs['bland'], neg_model.probs['bland'])
  print('worst:', pos_model.probs['worst'], neg_model.probs['worst'])
  print('2-star:', pos_model.probs['2-star'], neg_model.probs['2-star'])
  # print('delici:', log_ratios['delici'])
  # print('worst:', log_ratios['worst'])

  log_ratios = sorted(log_ratios.items(), key=lambda i: i[1], reverse=True)

  print('Top 20 log ratio:')
  print(log_ratios[:20])
  print('Bottom 20 log ratio:')
  print(log_ratios[-20:])

def naive_bayes(corpus, vocab):
  pos_count = len([d for d in corpus if d.label])
  neg_count = len(corpus) - pos_count
  pos_prob = pos_count / len(corpus)
  neg_prob = neg_count / len(corpus)
  log_ratio = math.log(pos_prob / neg_prob)

  # rank_nb_log_ratio(pos_model, neg_model)

  if os.path.isfile('./test_corpus.pickle'):
    print('load save ./test_corpus.pickle')

    with open('./test_corpus.pickle','rb') as pf:
      test_corpus = pickle.load(pf)
  else:
    test_corpus = read_folder('./yelp/test')
    for d in test_corpus:
      d.filter_vocab(vocab)

    with open('./test_corpus.pickle','wb') as pf:
      pickle.dump(test_corpus, pf)

  deltas = [0.01, 0.1, 1, 10]

  for delta in deltas:
    pos_model, neg_model = build_lang_models(corpus, vocab, delta)

    f = lambda d: log_ratio + sum([
      math.log(pos_model.probs[w]) - math.log(neg_model.probs[w])
      for w in d.features
    ])

    results = sorted([(d, f(d)) for d in test_corpus], key=lambda i: i[1], reverse=True)

    results = [r[0] for r in results]

    curve = []
    pos_count = len([d for d in results if d.label])

    for i in range(len(results)):
      tp = len([d for d in results[:i + 1] if d.label])

      precision = tp / (i + 1)
      recall = tp / pos_count

      curve.append((precision, recall))

    plt.plot([p[1] for p in curve], [p[0] for p in curve])

  plt.grid()
  plt.xlabel('recall')
  plt.ylabel('precision')
  plt.title('Precision Recall Curve')
  plt.legend([
    'delta=0.01',
    'delta=0.1',
    'delta=1',
    'delta=10'
  ], loc='lower left')
  plt.show()

def calc_idf(corpus, vocab):
  df = Counter()
  for d in corpus:
    df.update(d.features.keys())

  return {w: 1 + math.log(len(corpus) / df[w]) for w in vocab}

def tf_idf(corpus, idf):
  vocab = sorted(idf.keys())

  print('init corpus feature vectors')
  for d in corpus:
    d.set_vector(vocab, idf)

def knn(corpus, vocab):
  idf = calc_idf(corpus, vocab)

  query = read_folder('./query')
  tf_idf(query, idf)

  print('fit KNN model')

  classifier = KNN(5, 5)
  classifier.fit([d.vector for d in corpus], corpus)

  start_time = time.time()

  for i, d in enumerate(query):
    print('Query Doc', i)
    print(d.tokens)

    # neighbors = classifier.brute_force(d.vector)
    neighbors = classifier.neighbors(d.vector)
    print('Query Neighbors', i)

    for n in neighbors:
      print(n.tokens)
      print('\n')

    print('\n')

  print("--- %s seconds ---" % (time.time() - start_time))




def main():
  if os.path.isfile(SAVE_PATH):
    print('load save', SAVE_PATH)

    with open(SAVE_PATH,'rb') as pf:
      corpus, vocab, idf = pickle.load(pf)
  else:
    corpus = read_folder('./yelp')
    vocab = select_features(corpus)

    print('Corpus size:', len(corpus))
    for d in corpus:
      d.filter_vocab(vocab)
    corpus = [d for d in corpus if len(d.features) > 5]
    print('Corpus size (feature > 5):', len(corpus))

    idf = calc_idf(corpus, vocab)
    tf_idf(corpus, idf)

    with open(SAVE_PATH, 'wb') as pf:
      pickle.dump((corpus, vocab), pf)

  # naive_bayes(corpus, vocab)
  knn(corpus, idf)

main()

