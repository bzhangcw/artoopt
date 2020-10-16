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


class ModelParam(object):

  def __init__(self, *args, **kwargs) -> None:
    pass

  def extern_obj(self, *args):
    return 0

  def extern_obj_lag(self, *args):
    return 0

  def extern_nabla_lag(self, *args):
    return 0


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
    """iteration using geometric series
    Returns:
        Bool: [whether the stepsize is changed]
    """
    self.a *= self.power
    return 1

  def iter_b(self, dl, g_star, g, unimp_iters=0):
    """iteration using:
     |g - g_star| * rho / ||∂l||
      if unimproved over maximum threshold, 
        reduce step size by * .5
    Args:
        dl ([type]): ∂l
        g_star ([type]): g^\star, literally
        g ([type]): current relaxed objective
        unimp_iters (int, optional): [ unimproved iterations ].
           Defaults to 0.

    Returns:
        [bool]: whether the stepsize is changed
    """
    flg = unimp_iters > self.UNIMP_ITER_MAX
    if flg:
      print(f"objective unimproved {unimp_iters}")
      self.rho *= 0.5
    self.a = (g - g_star) * self.rho / (dl.dot(dl))
    return flg


class LagrangeModel(object):

  def __init__(self, *args, **kwargs):

    self.l = None
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
    pass

  def obj_lag(self):
    pass

  def nabla_lag(self):
    pass

  def primal_sol(self):
    """function to solve the primal model
    """
    pass

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
