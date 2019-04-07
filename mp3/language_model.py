import random
import math
from decimal import Decimal

D = Decimal

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

    self.uni_sum = None # only for additive cache


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

  def calc_additive_smooth_prob(self, token):
    assert self.N == 1 # only allow unigram to use

    if token in self.ml_cache:
      return self.ml_cache[token]

    if self.uni_sum is None:
      self.uni_sum = sum(self.counts.values()) + self.delta * len(self.vocab)

    val = self.counts[token] if token in self.counts else 0
    val = (val + self.delta) / self.uni_sum

    self.ml_cache[token] = val

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
      return self.calc_additive_smooth_prob(tokens[0])

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
      return self.calc_additive_smooth_prob(tokens[0])

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

  def perplexity(self, review, smoothing='linear'):
    tokens = review.tokens

    # either one uses additive smoothing for unigram in the end
    calc_prob_name = 'calc_abs_discount_prob' if smoothing == 'absolute' else 'calc_linear_smooth_prob'

    if self.N > 1:
      calc_prob = getattr(self.ref, calc_prob_name)
      start_prob = calc_prob(*tokens[:self.N - 1])
    else:
      start_prob = D(1)

    likelihood = math.log(start_prob)
    calc_prob = getattr(self, calc_prob_name)

    # bigram starts with i = 2 which leads to chunk [0:2]
    for i in range(self.N, len(tokens) + 1):
      token_chunk = tokens[i - self.N:i]
      likelihood += math.log(calc_prob(*token_chunk))

    return math.exp(-likelihood / len(tokens))
