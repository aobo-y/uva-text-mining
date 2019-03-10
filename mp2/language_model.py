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

  def calc_prob(self, prefix_token, token):
    ''' Additive smoothed probability '''

    if prefix_token not in self.cache:
      self.cache[prefix_token] = {
        'sum': sum(self.counts[prefix_token].values()) + self.delta * len(self.tokens),
        'probs': {}
      }

    if token in self.cache[prefix_token]['probs']:
      return self.cache[token]['probs'][token]


    val = self.counts[token] if token in self.counts else 0
    val = (val + self.delta) / self.cache[prefix_token]['sum']

    self.cache[token]['probs'][token] = val

    return val

  def sampling(self, prefix_token):
    prob = D(random.random())

    for token in self.tokens:
      token_prob = self.calc_prob(prefix_token, token)
      prob -= token_prob
      if prob < 0:
        return token, token_prob

    raise Exception('Failed to sample: Out of vocab')
