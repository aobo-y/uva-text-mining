# Machine Problem 2

## Part1: Parameter Estimation

### 1.1

```python
def calc_prob(self, pre_token, token):
  ''' Additive smoothed probability '''

  counts = self.counts[pre_token]

  if pre_token not in self.cache:
    self.cache[pre_token] = {
      'sum': sum(counts.values()) + self.delta * len(self.tokens),
      'probs': {}
    }

  cache = self.cache[pre_token]

  if token in cache['probs']:
    return cache['probs'][token]

  val = counts[token] if token in counts else 0
  val = (val + self.delta) / cache['sum']

  cache['probs'][token] = val

  return val
```

### 1.2

Top 10 probable words under tag `NN` and tags after tag `VB`

idx |Words under `NN`|Tags after `VB`
-|-|-
1 | % | DT
2 | company | IN
3 | year | VBN
4 | market | JJ
5 | trading | NN
6 | stock | NNS
7 | program | PRP$
8 | president | RB
9 | share | TO
10 | government | PRP









