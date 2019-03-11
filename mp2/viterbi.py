from math import inf, log

def viterbi(trs_mdl, ems_mdl, seqs):
  '''
  Inputs:
    trs_mdl: transition language model
    ems_mdl: emission language model
    seqs: sequence of tokens
  Outputs:
    hseqs: hidden state sequence
    score: corresponding log likelihood
  '''

  hstates = trs_mdl.tokens # hidden states

  v_scores = [{t: -inf for t in hstates} for _ in range(len(seqs))]
  b_hstates = [{t: None for t in hstates} for _ in range(len(seqs))]

  for idx, token in enumerate(seqs):
    pre_states = hstates if idx > 0 else ['START']

    for state in hstates:
      for pre_state in pre_states:
        pre_score = v_scores[idx - 1][pre_state] if idx > 0 else 0

        trs_score = log(trs_mdl.calc_prob(pre_state, state))
        ems_score = log(ems_mdl.calc_prob(state, token))

        new_score = pre_score + trs_score + ems_score
        if new_score > v_scores[idx][state]:
          v_scores[idx][state] = new_score
          b_hstates[idx][state] = pre_state

  max_score = -inf
  last_state = None

  for state, score in v_scores[len(seqs) - 1].items():
    if score > max_score:
      max_score = score
      last_state = state

  hseqs = [None] * len(seqs)
  hseqs[-1] = last_state
  for i in range(len(seqs) - 1, 0, -1):
    hseqs[i - 1] = b_hstates[i][hseqs[i]]

  return hseqs, max_score
