import os
import json
import re
import pickle
import numbers
import random
from collections import Counter

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.snowball import EnglishStemmer


STOP_WORD_PATH = './stop_words.txt'

tokenizer = TreebankWordTokenizer()
stemmer = EnglishStemmer()

stop_words = None

def normalize(token):
  token = re.sub(r'(^\W+|\W+$)', '', token)
  token = token.lower()
  if re.fullmatch(r'\d+(,\d{3})*(\.\d+)?', token):
    token = 'NUM'

  return stemmer.stem(token.strip())


with open(STOP_WORD_PATH) as sw_file:
  stop_words = sw_file.read().split('\n')
  stop_words = {normalize(w) for w in stop_words if w != ''}

class Review:
  def __init__(self, obj):
    # self.author = obj["Author"]
    self.rid = obj['ReviewID']
    self.date = obj['Date']
    self.label = 0 if float(obj['Overall']) < 4 else 1
    content = obj['Content']

    tokens = [normalize(token) for token in tokenizer.tokenize(content)]
    tokens = [token for token in tokens if token not in stop_words and token != '']
    self.tokens = Counter(tokens)

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
