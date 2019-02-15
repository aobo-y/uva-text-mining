import os
import json
import re
import pickle
import numbers
import random
from decimal import Decimal
# from functools import reduce
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.snowball import EnglishStemmer

D = Decimal

SAVE_PATH = './data/lang_model_save.pickle'

tokenizer = TreebankWordTokenizer()
stemmer = EnglishStemmer()

# Utils
def in_nested_map(n_map, keys):
  temp = n_map
  for key in keys:
    if key not in temp:
      return False
    else:
      temp = temp[key]

  return True

def get_nested_map(n_map, keys):
  temp = n_map
  for key in keys:
    if key not in temp:
      return None
    else:
      temp = temp[key]

  return temp

def update_nested_map(n_map, keys, val):
  temp = n_map
  for (idx, key) in enumerate(keys):
    if key not in temp:
      if idx == len(keys) - 1: # last ele
        temp[key] = val
      else:
        temp[key] = {}

    temp = temp[key]




def normalize(token):
  token = re.sub(r'(^\W+|\W+$)', '', token)
  token = token.lower()
  if re.fullmatch(r'\d+(,\d{3})*(\.\d+)?', token):
    token = 'NUM'

  return stemmer.stem(token.strip())


class Review:
  def __init__(self, obj):
    # self.author = obj["Author"]
    self.rid = obj["ReviewID"]
    self.date = obj["Date"]
    content = obj["Content"]

    tokens = [normalize(t) for t in tokenizer.tokenize(content)]
    self.tokens = [t for t in tokens if t != '']


class LanguageModel:

  def __init__(self, N, counts, vocab, ref=None):
    self.lamda = D(0.9)
    self.delta =  D(0.1)

    self.N = N
    self.counts = counts

    if N > 1:
      assert ref != None # require ref model except for unigram

    self.ref = ref
    self.ml_cache = {} # to cahce to on-the-fly calculated ml probs
    self.vocab = vocab


  def get_count(self, *tokens):
    return get_nested_map(self.counts, tokens)


  def calc_ml_prob(self, *tokens):
    assert len(tokens) == self.N

    if in_nested_map(self.ml_cache, tokens):
      return get_nested_map(self.ml_cache, tokens)

    prefix_counts = get_nested_map(self.counts, tokens[:-1])
    if prefix_counts is None:
      return None

    val = prefix_counts[tokens[-1]] if tokens[-1] in prefix_counts else 0
    val = D(val) / D(sum(prefix_counts.values()))

    update_nested_map(self.ml_cache, tokens, val)

    return val


  def calc_linear_smooth_prob(self, *tokens):
    assert len(tokens) == self.N # tokens len should match the N-gram model

    if self.N > 1:
      ml_prob = self.calc_ml_prob(*tokens)

      # have not encounter this prefix, fully back off
      if ml_prob is None:
        return self.ref.calc_linear_smooth_prob(*tokens[1:])

      return self.lamda * ml_prob + (1 - self.lamda) * self.ref.calc_linear_smooth_prob(*tokens[1:])

    else:
      return self.calc_ml_prob(*tokens)

  def calc_abs_discount_prob(self, *tokens):
    assert len(tokens) == self.N # tokens len should match the N-gram model

    if self.N > 1:
      prefix_counts = self.get_count(*tokens[:-1])

      # have not encounter this prefix, fully back off
      if prefix_counts is None:
        return self.ref.calc_abs_discount_prob(*tokens[1:])

      count = prefix_counts[tokens[-1]] if tokens[-1] in prefix_counts else 0
      S = len(prefix_counts.keys())
      prefix_sum = sum(prefix_counts.values())

      return max((count - self.delta), D(0)) / prefix_sum + (self.delta * S / prefix_sum) * self.ref.calc_abs_discount_prob(*tokens[1:])

    else:
      return self.calc_ml_prob(*tokens)

  def sampling(self, *prefix_tokens, smoothing='linear'):
    # either one is the same for unigram
    calc_prob = self.calc_abs_discount_prob if smoothing == 'absolute' else self.calc_linear_smooth_prob

    prob = D(random.random())

    for token in self.vocab:
      token_prob = calc_prob(*prefix_tokens, token)
      prob -= token_prob
      if prob < 0:
        return token, token_prob

    raise Exception('Failed to sample: Out of vocab')


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


def get_bigram_of(bi_model, pre_word):
  words = bi_model.counts[pre_word].keys()

  l_results = [(word, bi_model.calc_linear_smooth_prob(pre_word, word)) for word in words]
  a_results = [(word, bi_model.calc_abs_discount_prob(pre_word, word)) for word in words]

  l_results.sort(reverse=True, key=lambda t: t[1])
  a_results.sort(reverse=True, key=lambda t: t[1])

  print('linear:')
  for t in l_results[:10]:
    print(t[0], t[1])

  print('\nabs discount')
  for t in a_results[:10]:
    print(t[0], t[1])


def sample_sentences(uni_model, bi_model):
  print('[unigram]:')
  for _ in range(10):
    tokens = []
    likelihood = 1;

    for _ in range(15):
      token, prob = uni_model.sampling()
      tokens.append(token)
      likelihood *= prob

    print(' '.join(tokens), '{:.5g}'.format(likelihood))

  for smoothing in ['linear', 'absolute']:
    print('[bigram]', smoothing + ':')

    for _ in range(10):
      token, porb = uni_model.sampling()
      tokens = [token]
      likelihood = porb;

      for _ in range(14):
        token, prob = bi_model.sampling(token, smoothing=smoothing)
        tokens.append(token)
        likelihood *= prob

      print(' '.join(tokens), '{:.5g}'.format(likelihood))


def main():
  if os.path.isfile(SAVE_PATH):
    print('load save', SAVE_PATH)

    with open(SAVE_PATH,'rb') as pf:
      reviews, uni_counts, bi_counts = pickle.load(pf)
  else:
    reviews = read_folder('./yelp/train')
    uni_counts, bi_counts = count_grams(reviews)

    with open(SAVE_PATH, 'wb') as pf:
      pickle.dump((reviews, uni_counts, bi_counts), pf)

  vocab = list(uni_counts.keys())
  uni_model = LanguageModel(1, uni_counts, vocab)
  bi_model = LanguageModel(2, bi_counts, vocab, uni_model)

  # get_bigram_of(bi_model, 'good')
  sample_sentences(uni_model, bi_model)

main()

