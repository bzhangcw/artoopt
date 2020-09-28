# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /qap_utils.py
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 5:09:55 pm
# @description:
import logging
import os
import pickle as pk
import sys
from logging.handlers import TimedRotatingFileHandler as TRFH

import mosek.fusion as mf
from mosek import callbackcode, dinfitem, iinfitem, liinfitem

from ..conf import *
from ..deserialize_qapdata import *
from ..qap_georounding import *

expr = mf.Expr
dom = mf.Domain
mat = mf.Matrix
LOG_PATH = 'log'
FORMAT = '[%(name)s:%(levelname)s] [%(asctime)s] %(message)s'
logging.basicConfig(format=FORMAT)
if LOG_PATH is not None:
  if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
handler = TRFH(
    f'{LOG_PATH}/qap.log', when='M', interval=1, backupCount=7, encoding='utf8')
handler.setFormatter(logging.Formatter(FORMAT))
log = logging.getLogger()
log.addHandler(handler)
log.setLevel(logging.INFO)


class QAPParam(object):

  def __init__(
      self,
      A0,  # matrix A
      B0,  # matrix B
      obj,  # known best objective
      arr,  # known best solution
      scaling='l1',
      **kwargs):
    self.A0, self.B0 = A0, B0
    n, m = A0.shape
    self.n = n
    self.m = m
    self.best_obj = obj
    self.xo = np.zeros((n, n))

    for idx, v in enumerate(arr):
      self.xo[idx, v - 1] = 1

    e = np.ones(shape=n)
    E = np.ones(shape=(n, n))

    # scale the input matrix
    if scaling is None:
      A, B = A0, B0
    else:
      _scl = scaling.lower()
      if _scl == 'l1':
        A = A0 / A0.max()
        B = B0 / B0.max()
      else:
        A, B = A0, B0

    ab = np.kron(B.T, A.T)
    self.A = A
    self.B = B
    self.e = e
    self.E = E
    self.ab = ab


class QAPDerivative(object):

  def __init__(self, param: QAPParam = None, *args):
    if param is None:
      self.A, self.B, self.n, self.m, self.e, self.E, self.ab = args
    else:
      self.A, self.B, self.n, self.m, self.e, self.E, self.ab \
        = param.A, param.B, param.n, param.m, param.e, param.E, param.ab

  def partial_f(self, X):
    """derivative of QAP objective
      '\nabla F_\mu  = A^TXB + AXB^T'
    Args:
        X ([type]): [description]
    """
    return self.A.T.dot(X).dot(self.B) + self.A.dot(X).dot(self.B.T)

  def obj(self, X):
    return self.A.T.dot(X).dot(self.B).dot(X.T).trace()


class QAPTest(object):

  def __init__(self, name='', model=None, *args, **kwargs):
    self.name = name
    self.model = model
    self.args = args
    self.kwargs = kwargs


def check_obj_val(param, x_sol):
  _obj = param.A0.T.dot(x_sol).dot(param.B0).dot(x_sol.T).trace()
  return _obj


def set_mosek_model_params(model, **kwargs):
  """set template parameters
  """
  # unpacking
  mioMaxTime = kwargs.get('mioMaxTime', 60)

  # settings
  model.setLogHandler(sys.stdout)
  model.setSolverParam("mioMaxTime", mioMaxTime)
  model.acceptedSolutionStatus(mf.AccSolutionStatus.Anything)
  userCallback = makeUserCallback(model=model)
  model.setDataCallbackHandler(userCallback)
  return userCallback


def makeUserCallback(model):

  def userCallback(caller, douinf, intinf, lintinf):
    if caller == callbackcode.new_int_mio:
      print(f"new mip primal solution found @{douinf[dinfitem.optimizer_time]}")
    else:
      pass
    return 0

  return userCallback
