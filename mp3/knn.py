import numpy as np
import collections

sgn = lambda v: 1 if v >= 0 else 0

def cosine_similarity(v1, v2):
  return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class KNN:
  def __init__(self, n_hashbits, n_neighbors):
    self.n_hashbits = n_hashbits
    self.n_neighbors = n_neighbors

    self.hash_vectors = None
    self.buckets = None

  def init_hash_vectors(self, dimensions):
    self.hash_vectors = np.random.uniform(-1, 1, (self.n_hashbits, dimensions))

  def project(self, vector):
    hash_bits = self.hash_vectors.dot(vector)
    hash_bits = [sgn(b) for b in hash_bits]
    hash_bits = ''.join(str(b) for b in hash_bits)

    return int(hash_bits, 2)

  def fit(self, vectors, labels):
    # vectors should have same dimension size
    self.init_hash_vectors(len(vectors[0]))
    self.buckets = collections.defaultdict(list)

    for v, l in zip(vectors, labels):
      key = self.project(v)
      self.buckets[key].append((v, l))

  def predict(self, vector):
    neighbors = self.neighbors(vector)

    labels = collections.Counter(neighbors)
    return max(labels.items(), key=lambda i: i[1])[0]

  def neighbors(self, vector):
    key = self.project(vector)
    bucket = self.buckets[key]

    results = [
      (cosine_similarity(v, vector), l)
      for v, l in bucket
    ]

    results = sorted(results, key=lambda r: r[0], reverse=True)
    if len(results) > self.n_neighbors:
      results = results[:self.n_neighbors]

    return [r[1] for r in results]

  def brute_force(self, vector):
    results = []

    for bucket in self.buckets.values():
      results += [
        (cosine_similarity(v, vector), l)
        for v, l in bucket
      ]

    results = sorted(results, key=lambda r: r[0], reverse=True)
    if len(results) > self.n_neighbors:
      results = results[:self.n_neighbors]

    return [r[1] for r in results]
