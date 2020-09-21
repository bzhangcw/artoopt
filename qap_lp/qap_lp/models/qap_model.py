# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /qap_models.py
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 2:14:31 pm
# @description:

from .qap_model_l2 import *
from .qap_utils import *




def check_obj_val(x_sol):
  _obj = A0.T.dot(x_sol).dot(B0).dot(x_sol.T).trace()
  print(f'original obj {_obj}')
  return _obj


def co_gm(param, rd=True, run_time=60):

  A, B, n, m, e, E, ab = param.A, param.B, param.n, param.m, param.e, param.E, param.ab
  S = np.block([[np.zeros((m, m)), 0.5 * np.eye(m)],
                [0.5 * np.eye(m), np.zeros((m, m))]]) + 1 * np.eye(2 * n)
  K = np.linalg.cholesky(S)
  model = mf.Model('qap')

  if rd:
    X = model.variable([*A.shape], dom.inRange(0, 1))
  else:
    X = model.variable("x", [*A.shape], dom.binary())
  Z = model.variable([*A.shape], dom.greaterThan(0.0))
  M = model.variable([*A.shape], dom.greaterThan(0.0))
  v = model.variable(1, dom.greaterThan(0.0))
  model.constraint(expr.sum(X, 0), dom.equalsTo(1))
  model.constraint(expr.sum(X, 1), dom.equalsTo(1))
  model.constraint(expr.sub(M, expr.mul(A, X)), dom.equalsTo(0))
  model.constraint(expr.sub(Z, expr.mul(X, B)), dom.equalsTo(0))

  x = expr.flatten(expr.mul(K, expr.vstack(Z, M)))
  model.constraint(expr.vstack(v, x), dom.inQCone())
  model.objective(mf.ObjectiveSense.Minimize, v)

  # set params
  model.setLogHandler(sys.stdout)
  model.setSolverParam("mioMaxTime", run_time)
  model.acceptedSolutionStatus(mf.AccSolutionStatus.Anything)
  userCallback = makeUserCallback(model=model, X=X)
  model.setDataCallbackHandler(userCallback)

  model.solve()

  model.flushSolutions()
  X_sol = X.level().reshape(A.shape)
  if rd:
    x, _ = extract_sol_rounding(X_sol, A, B)
    return x
  return X_sol


def makeUserCallback(model, X):

  def userCallback(caller, douinf, intinf, lintinf):
    if caller == callbackcode.new_int_mio:
      print(f"new mip primal solution found @{douinf[dinfitem.optimizer_time]}")
    else:
      pass
    return 0

  return userCallback
