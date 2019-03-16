import os
import re

def normalize(token):
  token = token.lower()
  # 7\/8/CD
  token = token.replace('\\/', '/')
  # if re.fullmatch(r'\d+(,\d{3})*(\.\d+)?', token) or \
  #   re.fullmatch(r'\d+[/|\-|:]\d+', token):
  #   token = 'NUM'

  return token.strip()

class Sentence:
  def __init__(self, words, tags):
    self.words = words
    self.tags = tags

def read_folder(path):
  sentences = []

  for fname in os.listdir(path):
    fpath = os.path.join(path, fname)

    if os.path.isfile(fpath):
      with open(fpath) as f:
        lines = f.read().split('\n')
        lines = [
          l.strip(' ').strip('=').lstrip('[').rstrip(']')
          for l in lines
        ]
        lines = [l for l in lines if l != '']

        words, tags = [], []
        for line in lines:
          line = line
          tokens = line.split(' ')

          # 7\/8/CD
          tokens = [t.rsplit('/', 1) for t in tokens if t != '']

          words += [normalize(t[0]) for t in tokens]
          tags += [
            t[1].split('|') if '|' in t[1] else t[1]
            for t in tokens
          ]

          if words[-1] == '.':
            sentences.append(Sentence(words, tags))
            words, tags = [], []


    else:
      sentences += read_folder(fpath)

  print('read', len(sentences), 'sentences from', path)

  return sentences
