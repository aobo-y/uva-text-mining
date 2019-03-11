import random
from decimal import Decimal

D = Decimal

class LanguageModel:

  def __init__(self, counts, tokens, delta=0.1):
    self.delta =  D(delta)

    self.counts = counts
    self.tokens = tokens

    # on the fly cache
    self.cache = {}

  def calc_prob(self, pre_token, token):
    ''' Additive smoothed probability '''

    assert token in self.tokens

    counts = self.counts[pre_token]

    if pre_token not in self.cache:
      self.cache[pre_token] = {
        'sum': sum(counts.values()) + self.delta * len(self.tokens),
        'probs': {}
      }

    cache = self.cache[pre_token]

    if token in cache['probs']:
      return cache['probs'][token]

    val = counts[token] if token in counts else 0
    val = (val + self.delta) / cache['sum']

    cache['probs'][token] = val

    return val

  def sampling(self, pre_token):
    prob = D(random.random())

    for token in self.tokens:
      token_prob = self.calc_prob(pre_token, token)
      prob -= token_prob
      if prob < 0:
        return token, token_prob

    raise Exception('Failed to sample: Out of vocab')
