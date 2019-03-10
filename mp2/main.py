from data_util import read_folder


def count_grams(sentences):
  trs_counts = {}
  ems_counts = {}

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
      for t in tag:
        add_counts(ems_counts, t, word)

      prev_tag = tag


  return trs_counts, ems_counts

def main():
  sentences = read_folder('./tagged')
  trs_counts, ems_counts = count_grams(sentences)
  # print(trs_counts)
  # print(ems_counts)

main()
