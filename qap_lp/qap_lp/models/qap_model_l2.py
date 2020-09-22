# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /qap_model_l2
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 3:06:38 pm
# @description:

from .qap_utils import *


def l2_naive(mu, param=None, rd=False, **kwargs):
  A, B, n, m, e, E, ab = param.A, param.B, param.n, param.m, param.e, param.E, param.ab
  # do Cholesky
  n = A.shape[0]
  P = np.kron(B.T, A.T) + mu * np.eye(n * n)
  U = np.linalg.cholesky(P)

  model = mf.Model('qap')

  if rd:
    X = model.variable([*A.shape], dom.inRange(0, 1))
  else:
    X = model.variable("x", [*A.shape], dom.binary())
  m = expr.mul(U, expr.flatten(X))
  v = model.variable(1, dom.greaterThan(0.0))
  model.constraint(expr.sum(X, 0), dom.equalsTo(1))
  model.constraint(expr.sum(X, 1), dom.equalsTo(1))
  model.constraint(expr.vstack(v, m), dom.inQCone())
  model.objective(mf.ObjectiveSense.Minimize, v)

  # set params
  userCallback = set_mosek_model_params(model, **kwargs)

  model.solve()

  model.flushSolutions()
  X_sol = X.level().reshape(A.shape)
  if rd:
    x, _ = extract_sol_rounding(X_sol, A, B)
    return x
  return X_sol


def l2_conic_georound(param, rd=True, **kwargs):

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
  userCallback = set_mosek_model_params(model, **kwargs)

  model.solve()

  model.flushSolutions()
  X_sol = X.level().reshape(A.shape)
  if rd:
    x, _ = extract_sol_rounding(X_sol, A, B)
    return x
  return X_sol