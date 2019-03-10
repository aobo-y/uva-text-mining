# Machine Problem 2

## Part1: Parameter Estimation

### 1.1

```python
def calc_prob(self, prefix_token, token):
  ''' Additive smoothed probability '''

  if prefix_token not in self.cache:
    self.cache[prefix_token] = {
      'sum': sum(self.counts[prefix_token].values()) + self.delta * len(self.tokens),
      'probs': {}
    }

  if token in self.cache[prefix_token]['probs']:
    return self.cache[token]['probs'][token]

  val = self.counts[token] if token in self.counts else 0
  val = (val + self.delta) / self.cache[prefix_token]['sum']

  self.cache[token]['probs'][token] = val

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









