# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /qap_model_l2
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 3:06:38 pm
# @description:
# L2 based formulations for QAP

from .qap_utils import *
from .qap_gradient_proj import *


def l2_naive(mu, param=None, rd=False, **kwargs):
  """naive formulation
      min_X f = tr(A'XBX')+μ⋅tr(XX')

  Args:
      mu (float): scaling parameter
      param (QAPParam, optional):. Defaults to None.
      rd (bool, optional): use geometric rounding, if True, the Mosek
        model relaxes the integral constraints. Defaults to False.

  Returns:
      X_sol: solution
  """
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
  """a better "naive" formulation, 
      min tr(M'SM), and R'R = S+δI, M = (XB, AX)'
​	
  Args:
      mu (float): scaling parameter
      param (QAPParam, optional):. Defaults to None.
      rd (bool, optional): use geometric rounding, if True, the Mosek
        model relaxes the integral constraints. Defaults to False.

  Returns:
      X_sol: solution
  """
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


def l2_exact_penalty_gradient_proj(param, **kwargs):
  """Exact penalty + Trace relaxation
      min_X f = tr(A'XBX') + μ⋅|tr(XX') − n|
    solved by gradient projection method
  """

  A, B, n, m, e, E, ab = \
    param.A, param.B, param.n, param.m, param.e, param.E, param.ab

  # hyper parameters
  mu = kwargs.get('mu', 0.1)

  # known best
  xo = param.xo
  opt = param.best_obj

  # initialize
  x = x0 = np.ones((n, n)) / n

  # 𝛁F
  nabla = QAPDerivativeL2Penalty(param, mu)

  x_sol = run_gradient_projection(x, param, nabla, **kwargs)
  return x_sol


if __name__ == "__main__":
  kwargs = {}
  instance_name = 'esc16h'
  # mosek params
  msk_params = {**MSK_DEFAULT, **kwargs}
  qap_params = {**QAP_DEFAULT, **kwargs}

  # coefficients
  A0, B0 = parse(f'{QAP_INSTANCE}/{instance_name}.dat')

  # parse known solution
  _, best_obj, arr = parse_sol(f'{QAP_SOL}/{instance_name}.sln')

  param = QAPParam(A0, B0, best_obj, arr, **qap_params)