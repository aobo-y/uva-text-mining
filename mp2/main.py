import math
from random import shuffle
from statistics import mean, stdev

from data_util import read_folder
from language_model import LanguageModel
from viterbi import viterbi

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

  return trs_counts, ems_counts, tag_set, word_set


def top_tokens(counts, num):
  counts = sorted(counts.items(), key=lambda i: i[1], reverse=True)
  return [c[0] for c in counts[:num]]


def build_models(sentences, delta=0.5, sigma=0.1):
  trs_counts, ems_counts, tags, words = count_grams(sentences)

  # print('Number of tags:', len(tags))
  # print('Number of words:', len(words))

  trs_model = LanguageModel(trs_counts, tags, delta)
  ems_model = LanguageModel(ems_counts, words, sigma)

  return trs_model, ems_model

def split_data(sentences, i):
  # 5-fold cross validation
  chk_size = math.ceil(len(sentences) / 5)

  start_idx = chk_size * i
  end_idx = chk_size * (i + 1)

  training = sentences[:start_idx] + sentences[end_idx:]
  testing = sentences[start_idx:end_idx]

  return training, testing

def eval_metric(tags, preds, labels):
  total, accurate =  0, 0
  tp = {}
  fp = {}
  fn = {}

  def increment(dic, k):
    if k in dic:
      dic[k] += 1
    else:
      dic[k] = 1

  for p_seq, l_seq in zip(preds, labels):
    for p, ls in zip(p_seq, l_seq):
      if not isinstance(ls, list):
        ls = [ls]

      total += 1
      if p in ls:
        accurate += 1

      for l in ls:
        if l == p:
          increment(tp, p)
        else:
          increment(fp, p)
          increment(fn, l)

  accuracy = accurate / total

  precisions, recalls = {}, {}

  for t in tags:
    t_tp = tp[t] if t in tp else 0
    t_fp = fp[t] if t in fp else 0
    t_fn = fn[t] if t in fn else 0

    precisions[t] = t_tp / (t_tp + t_fp) if t_tp + t_fp != 0 else 0
    recalls[t] = t_tp / (t_tp + t_fn) if t_tp + t_fn != 0 else 0

  return accuracy, precisions, recalls


def get_tags(sentences):
  tags = set()
  for s in sentences:
    for ts in s.tags:
      if not isinstance(ts, list):
        ts = [ts]
      for t in ts:
        if t not in tags:
          tags.add(t)

  return tags

def tune_params(sentences):
  shuffle(sentences)

  tags = get_tags(sentences)

  for delta in [7, 5, 4, 3, 2, 1, 0.8]:
      for sigma in [0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.008]:
        print('Delta, Sigma:',  delta, sigma)

        accuracy_results = []
        precision_results = []
        recall_results = []

        for i in range(5):
          # print('Cross-validation:', i)
          training, testing = split_data(sentences, i)

          # print('Training size:', len(training))
          # print('Testing size:', len(testing))

          trs_model, ems_model = build_models(training, delta, sigma)

          map_unk = lambda s: [w if w in ems_model.tokens else 'UNK' for w in s]

          # testing = testing[:5]
          preds = [viterbi(trs_model, ems_model, map_unk(s.words))[0] for s in testing]

          labels = [s.tags for s in testing]

          accuracy, precisions, recalls =  eval_metric(tags, preds, labels)

          accuracy_results.append(accuracy)
          precision_results.append(precisions)
          recall_results.append(recalls)

        print('accuracy:', round(mean(accuracy_results), 4))

        precisions, recalls = {}, {}
        for t in precision_results[0].keys():
          precisions[t] = mean([p[t] for p in precision_results])
          recalls[t] = mean([r[t] for r in recall_results])

        print('precisions:', round(mean(precisions.values()), 4))
        print('recalls:', round(mean(recalls.values()), 4))

        print('NN:', round(precisions['NN'], 4), round(recalls['NN'], 4))
        print('VB:', round(precisions['VB'], 4), round(recalls['VB'], 4))
        print('JJ:', round(precisions['JJ'], 4), round(recalls['JJ'], 4))
        print('NNP:', round(precisions['NNP'], 4), round(recalls['NNP'], 4))

def main():
  sentences = read_folder('./tagged')

  # tune_params(sentences)

  trs_model, ems_model = build_models(sentences, 3, 0.04)

  samples = []

  while len(samples) < 100:
    tags = []

    prev_tag = 'START'
    score = 0
    while len(tags) < 10:
      tag, prob = trs_model.sampling(prev_tag)

      if tag != '.' or len(tags) == 9:
        tags.append(tag)
        prev_tag = tag
        score += math.log(prob)

    sentence = []
    for t in tags:
      word, prob = ems_model.sampling(t)
      score += math.log(prob)
      sentence.append(word)

    samples.append({
      'sentence': sentence,
      'score': score
    })

  samples = sorted(samples, key=lambda m: m['score'], reverse=True)

  for m in samples[:10]:
    print(' '.join(m['sentence']))
    print(m['score'])




  # print(top_tokens(ems_counts['NN'], 10))
  # print(top_tokens(trs_counts['VB'], 10))

main()
