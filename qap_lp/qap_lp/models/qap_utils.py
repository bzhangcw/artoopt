# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /qap_utils.py
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 5:09:55 pm
# @description:
import pickle as pk
import sys

import mosek.fusion as mf
from mosek import callbackcode, dinfitem, iinfitem, liinfitem

from qap_lp.deserialize_qapdata import *
from qap_lp.qap_georounding import *

expr = mf.Expr
dom = mf.Domain
mat = mf.Matrix


class QAPParam(object):

  def __init__(
      self,
      A,
      B,
      n,
      m,
      e,
      E,
      ab,
  ):
    self.A, self.B, self.n, self.m, self.e, self.E, self.ab = A, B, n, m, e, E, ab


class QAPDerivative(object):

  def __init__(
      self,
      A,
      B,
      n,
      m,
      e,
      E,
      ab,  # args
      param: QAPParam = None):
    if param is None:
      self.A, self.B, self.n, self.m, self.e, self.E, self.ab = A, B, n, m, e, E, ab
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


class QAPTest(object):

  def __init__(self, name='', model=None, *args, **kwargs):
    self.name = name
    self.model = model
    self.args = args
    self.kwargs = kwargs


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