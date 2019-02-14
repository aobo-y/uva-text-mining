import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def readfile(name):
  with open(name) as file:
    lines = file.read().split('\n')
    lines = [line.strip() for line in lines]
    lines = [line.split('\t') for line in lines if line != '']
    vals = np.array([int(line[1]) for line in lines])

  return vals


def plot(name):
  print('----' + name + '----')

  vals = readfile(name + '.txt')
  idx = np.array(range(1, len(vals) + 1))

  reg = LinearRegression().fit(np.log10(idx).reshape(-1, 1), np.log10(vals).reshape(-1, 1))

  coef = reg.coef_[0][0]
  intercept = reg.intercept_[0]
  print('coef', reg.coef_)
  print('intercept', reg.intercept_)

  line_range = np.array(range(6))
  line_val = 10 ** (line_range * coef + intercept)
  line_idx = 10 ** line_range

  plt.scatter(idx, vals)
  plt.plot(line_idx, line_val, color="red")
  plt.xscale('log')
  plt.yscale('log')
  plt.grid()

  plt.title(name.upper())

  plt.show()

plot('ttf')
plot('df')
