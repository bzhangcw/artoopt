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

  def extern_obj(self, x, y):
    return self.c.T.dot(y).trace() - self.f.dot(x)

  def extern_obj_lag(self, x, y, l):
    return self.obj(x, y) + l.sum() - l.T.dot(y).sum()

  def extern_nabla_lag(self, y):
    return self.em - y.sum(1)


class StepSize(object):
  # a pack of different methodologies
  # for stepsize computation
  UNIMP_ITER_MAX = 7

  def __init__(self, initial_stp, **kwargs):
    self.a = self.a0 = initial_stp
    self.power = kwargs.get('power', .9)
    self.rho = self.rho0 = kwargs.get("rho", 2)

  def reset(self):
    self.a = self.a0

  def iter_geometric(self):
    self.a *= self.power
    return 1

  def iter_b(self, dl, g_star, g, unimp_iters=0):
    flg = unimp_iters > self.UNIMP_ITER_MAX
    if flg:
      print(f"objective unimproved {unimp_iters}")
      self.rho *= 0.5
    self.a = (g - g_star) * self.rho / (dl.dot(dl))
    return flg

  # DIRECT
  def geometric_series(self, iteration=1, power=0.9):
    return self.a0 * (power**iteration)


class LagrangeModel(UFLParam):

  def __init__(self, m, n, c, f):
    super().__init__(m, n, c, f)

    self.l = np.ones(self.m)
    self.dl = None
    self.stp = StepSize(1, power=0.9, rho=2)
    self.val = 0
    self.val_lag = 0
    self.unimproved_flag = False
    self.unimproved_iter = 0
    # start at random
    self.rand_int()
    #
    self.g_star = self.val
    self.val_best = self.val
    self.gap = np.inf

  def rand_int(self):

    self.x = np.zeros(self.n)
    self.y = np.zeros((self.m, self.n))
    self.val_best = np.inf
    _j = np.random.randint(0, self.n)
    self.y[:, _j] = 1
    self.x[_j] = 1

    self.val = self.obj()
    self.val_lag = self.obj_lag()

  def obj(self):
    return self.c.T.dot(self.y).trace() - self.f.dot(self.x)

  def obj_lag(self):
    return self.obj() + self.l.sum() - self.l.T.dot(self.y).sum()

  def nabla_lag(self):
    return self.em - self.y.sum(1)

  def primal_sol(self):
    # for y
    x = np.zeros(self.n)
    y = np.zeros((self.m, self.n))
    c_l = np.zeros((self.m, self.n))

    for i in range(self.m):
      for j in range(self.n):
        c_l[i, j] = max(self.c[i, j] - self.l[i], 0)

    x = ((c_l.sum(0) - self.f) > 0).astype(float)
    for i in range(self.m):
      for j in range(self.n):
        y[i, j] = ((self.c[i, j] - self.l[i]) > 0) * x[j]

    self.x, self.y = x, y

    return x, y

  def eval(self):
    self.val = self.obj()
    val_lag = self.obj_lag()
    if val_lag - self.val_best < 0:
      self.val_best = val_lag

    _gap = val_lag - self.val_best

    if _gap < self.gap:
      self.unimproved_flag = False
      self.unimproved_iter = 0
      self.gap = _gap
    else:
      self.unimproved_flag = True
      self.unimproved_iter += 1

    # print(self.gap, _gap)
    # if val_lag < self.val_lag:
    #   self.unimproved_flag = False
    #   self.unimproved_iter = 0
    # else:
    #   self.unimproved_flag = True
    #   self.unimproved_iter += 1

    self.val_lag = val_lag

  def iter(self, step_size_method='b'):
    """iteration of the sub-gradient method
    """
    # self.stp.iter_geometric()

    _ = self.primal_sol()
    self.eval()
    self.dl = self.nabla_lag()
    if np.abs(self.dl).max() < 1e-3:
      return 0
    if step_size_method == 'geo':
      stp_shrink = self.stp.iter_geometric()
    elif step_size_method == 'b':
      stp_shrink = self.stp.iter_b(self.dl, self.g_star, self.val_lag,
                                   self.unimproved_iter)
    else:
      raise ValueError("no such stepsize method")

    if stp_shrink:
      self.unimproved_iter = 0
    if self.stp.a < 1e-6:
      return 0

    self.l -= self.stp.a * self.dl
    if np.abs(self.dl).max() < 1e-3:
      return 0
    print(f"f: {self.val}, f_lag: {self.val_lag}")

    return 1

  def run(self, max_iter=1000, step_size_method='b'):
    for i in range(max_iter):
      flg = self.iter(step_size_method)
      if not flg:
        print(f"finished @iteration {i}")
        break


if __name__ == "__main__":
  m, n = 200, 21
  c = np.random.randint(1, 10, size=(m, n))
  f = np.random.randint(20, 100, size=(n))
  md = LagrangeModel(m, n, c, f)
  # md.run()
