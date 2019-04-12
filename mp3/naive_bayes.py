import math

from language_model import LanguageModel
from collections import Counter


# BernoulliNB
class NaiveBayes:
  def __init__(self, corpus, vocab, delta = 0.1):
    self.corpus = corpus
    self.vocab = vocab

    pos_count = len([d for d in corpus if d.label])
    neg_count = len(corpus) - pos_count
    pos_prob = pos_count / len(corpus)
    neg_prob = neg_count / len(corpus)
    self.log_ratio = math.log(pos_prob / neg_prob)

    self.build_lang_models(delta)

  def build_lang_models(self, delta = 0.1):
    pos_counts = Counter()
    neg_counts = Counter()

    for review in self.corpus:
      if review.label:
        pos_counts.update(review.features)
      else:
        neg_counts.update(review.features)

    self.pos_model = LanguageModel(pos_counts, self.vocab, delta)
    self.neg_model = LanguageModel(neg_counts, self.vocab, delta)

  def predict(self, review):
    return 1 if self.f_value(review) >= 0 else 0

  def f_value(self, review):
    return self.log_ratio + sum([
      c * (math.log(self.pos_model.probs[w]) - math.log(self.neg_model.probs[w]))
      for w, c in review.features.items()
    ])
