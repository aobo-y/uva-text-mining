from data_util import read_folder
from language_model import LanguageModel

def count_grams(sentences):
  trs_counts = {}
  ems_counts = {}

  tag_set = set()
  word_set = set()

  def add_counts(counts, prev_token, token):
    if prev_token in counts:
      if token in counts[prev_token]:
        counts[prev_token][token] += 1
      else:
        counts[prev_token][token] = 1
    else:
      counts[prev_token] = {token: 1}

  for s in sentences:
    prev_tag = 'START'

    for tag, word in zip(s.tags, s.words):
      # may have multiple tag
      if not isinstance(prev_tag, list):
        prev_tag = [prev_tag]
      if not isinstance(tag, list):
        tag = [tag]

      for p_t in prev_tag:
        for t in tag:
          add_counts(trs_counts, p_t, t)
          tag_set.add(t)
      for t in tag:
        add_counts(ems_counts, t, word)
        word_set.add(word)

      prev_tag = tag

  word_set.add('UNK')

  return trs_counts, ems_counts, list(tag_set), list(word_set)


def top_tokens(counts, num):
  counts = sorted(counts.items(), key=lambda i: i[1], reverse=True)
  return [c[0] for c in counts[:num]]

def main():
  sentences = read_folder('./tagged')
  trs_counts, ems_counts, tags, words = count_grams(sentences)

  print('Number of tags:', len(tags))
  print('Number of words:', len(words))

  trs_model = LanguageModel(trs_counts, tags, delta=0.5)
  ems_model = LanguageModel(ems_counts, words, delta=0.1)

  print(top_tokens(ems_counts['NN'], 10))
  print(top_tokens(trs_counts['VB'], 10))
main()
