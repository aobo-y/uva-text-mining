# import random
# import math


class LanguageModel:
  def __init__(self, counts, vocab, delta=0.1):
    self.delta = delta

    self.counts = counts

    self.vocab = vocab

    self.probs = {}
    self.calc_additive_smooth_prob()


  def calc_additive_smooth_prob(self):
    total = sum(self.counts.values()) + self.delta * len(self.vocab)

    for word in self.vocab:
      self.probs[word] = (self.counts[word] + self.delta) / total

