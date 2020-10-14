# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: ufllag
# @file: /main.py
# @created: Wednesday, 14th October 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Wednesday, 14th October 2020 3:16:11 pm
# @description:
import numpy as np
import sys

np.random.seed(1)


class UFLParam(object):

  def __init__(self, m, n, c, f) -> None:
    self.m, self.n, self.c, self.f = m, n, c, f
    self.e = np.ones((n, 1))
    self.en = np.ones(n)
    self.em = np.ones(m)

  def obj(self, x, y):
    return self.c.T.dot(y).trace() - self.f.dot(x)

  def obj_lag(self, x, y, l):
    return self.obj(x, y) + l.sum() - l.T.dot(y).sum()

  def nabla_lag(self, y):
    return self.em - y.sum(1)


def primal_sol(param, l):
  # for y
  x = np.zeros(param.n)
  y = np.zeros((param.m, param.n))
  c_l = np.zeros((param.m, param.n))
  for i in range(param.m):
    for j in range(param.n):
      c_l[i, j] = max(param.c[i, j] - l[i], 0)

  x = ((c_l.sum(0) - param.f) > 0).astype(float)
  for i in range(param.m):
    for j in range(param.n):
      y[i, j] = ((param.c[i, j] - l[i]) > 0) * x[j]
  return x, y


def model(param: UFLParam):
  l = np.ones(param.m)
  stp = 1
  for i in range(1000):
    stp *= 0.9
    x, y = primal_sol(param, l)
    print(param.obj(x, y), param.obj_lag(x, y, l))
    dl = param.nabla_lag(y)
    l -= stp * dl
    if stp < 1e-4:
      break
  return 0


def main(m, n):
  c = np.random.randint(1, 10, size=(m, n))
  f = np.random.randint(50, 100, size=(n))
  param = UFLParam(m, n, c, f)
  md = model(param)


if __name__ == "__main__":
  main(50, 5)
