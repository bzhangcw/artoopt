# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: models
# @file: /qap_gradient_proj.py
# @created: Sunday, 27th September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Sunday, 27th September 2020 4:46:00 pm
# @description:

from .qap_utils import *

logger = logging.getLogger('qap.run.gradient_projection')


class QAPDerivativeL2Penalty(QAPDerivative):

  def __init__(self, param, mu, *args):
    super().__init__(param=param, *args)
    self.mu = mu

  def partial_f(self, X):
    df = super().partial_f(X)
    # return df - 2*
    return df - 2 * self.mu * X

  def obj(self, X):
    obj = super().obj(X)
    return obj - self.mu * X.dot(X.T).trace() + self.mu * X.shape[0]

  def original_obj(self, X):
    return super().obj(X)


def msk_pd_on_dc(
    param: QAPParam,
    dF,  # gradient: \nabla dF
    lb_indices,
    ub_indices=None,
    **kwargs):
  """The Mosek model to compute the projected gradient 
			onto orthogonal constraints.
		 The model:
			min ||D + F||_F
	Args:
			param (QAPParam): QAP params
			dF: gradient, dF
			lb_indices: the indices of `active` lower bound 
				inequality constraints
			ub_indices: not used
	Returns:
			solution, model, variable, and lb constraints
	"""

  A, B, n, m, e, E, ab = param.A, param.B, param.n, param.m, param.e, param.E, param.ab

  model = mf.Model('projected_gradient_on_D_cone')
  D = model.variable("d", [*A.shape], dom.unbounded())
  v = model.variable(1, dom.greaterThan(0))
  m = expr.flatten(expr.add(D, dF))

  model.objective(mf.ObjectiveSense.Minimize, v)

  model.constraint(expr.vstack(v, m), dom.inQCone())
  constrs_a = model.constraint(expr.sum(D, 0), dom.equalsTo(0))
  constrs_b = model.constraint(expr.sum(D, 1), dom.equalsTo(0))
  constrs_lb = model.constraint(D.pick(*lb_indices), dom.equalsTo(0))
  # constrs_ub = model.constraint(D.pick(*ub_indices), dom.lessThan(0))

  # set params
  userCallback = set_mosek_model_params(model, **kwargs)
  model.setLogHandler(None)
  model.solve()

  model.flushSolutions()
  D_sol = D.level().reshape(A.shape)
  return D_sol, model, D, constrs_lb, constrs_a, constrs_b


def msk_pd_on_dc_lp(
    param: QAPParam,
    dF,  # gradient: \nabla dF
    lb_indices,
    ub_indices=None,
    **kwargs):
  """The Mosek model to compute the projected gradient 
            onto orthogonal constraints. 
            this is the linear programming model, i.e.
         The model:
            min <D, F>
    Args:
            param (QAPParam): QAP params
            dF: gradient, dF
            lb_indices: the indices of `active` lower bound 
                inequality constraints
            ub_indices: not used
    Returns:
            solution, model, variable, and lb constraints
    """

  A, B, n, m, e, E, ab = param.A, param.B, param.n, param.m, param.e, param.E, param.ab

  model = mf.Model('projected_gradient_on_D_cone')
  D = model.variable("d", [*A.shape], dom.unbounded())
  m = expr.sum(expr.mulElm(D, dF))

  model.objective(mf.ObjectiveSense.Minimize, m)

  constrs_a = model.constraint(expr.sum(D, 0), dom.equalsTo(0))
  constrs_b = model.constraint(expr.sum(D, 1), dom.equalsTo(0))
  constrs_lb = model.constraint(D.pick(*lb_indices), dom.equalsTo(0))
  # constrs_ub = model.constraint(D.pick(*ub_indices), dom.lessThan(0))

  # set params
  userCallback = set_mosek_model_params(model, **kwargs)
  model.setLogHandler(None)
  model.solve()

  model.flushSolutions()
  D_sol = D.level().reshape(A.shape)
  return D_sol, model, D, constrs_lb, constrs_a, constrs_b


def msk_st(dp, x, param):
  """The Mosek model to compute the maximum stepsize 
			of the line search

	Args:
			dp: given computed gradient
			x: current point

	Returns:
			float: maximum stepsize
	"""
  n = param.n
  model = mf.Model('step_size')
  v = model.variable(1, dom.greaterThan(0))
  V = expr.reshape(expr.repeat(v, n * n, 0), [n, n])
  delta = expr.mulElm(V, dp)
  model.constraint(expr.add(x, delta), dom.greaterThan(0))
  model.objective(mf.ObjectiveSense.Maximize, v)
  #     model.setLogHandler(sys.stdout)
  try:
    model.solve()
    return v.level()[0]
  except:
    return 0


def run_gradient_projection(x, param: QAPParam, nabla: QAPDerivative, **kwargs):
  # unpacking solver parameters
  max_iter = kwargs.get('max_iteration', 1000)
  gd_method = kwargs.get('gd_method', msk_pd_on_dc)
  st_method = kwargs.get('st_method', msk_st)
  st_line_grids = kwargs.get('st_line_grids', 10)
  logging_interval = kwargs.get('logging_interval', 1)

  # unpacking params
  n = param.n
  xo = param.xo
  best_obj = nabla.obj(xo)

  # start iterations
  for i in range(max_iter):

    _obj = nabla.obj(x)
    _logging = i % logging_interval == 0
    _ac = False
    d0 = nabla.partial_f(x)

    # indices of active lower bound constraints
    lb_x, lb_y = np.where(x <= 1e-4)
    lb_x = lb_x.tolist()
    lb_y = lb_y.tolist()

    if _logging:
      logger.info(f'=====iteration: {i}====')

    # do projection
    while True:
      # compute gradient projection
      dp, m, D, constrs_lb, constrs_a, constrs_b = gd_method(
          param,
          d0,
          (lb_x, lb_y),
      )

      # evaluate norm of the projected gradient
      ndf = np.abs(dp).max()

      # fetch maximum stepsize
      stp = st_method(dp, x, param)

      # active set tuning if ||P(dF)|| < eps
      if ndf <= 1e-6 and stp <= 1e-6:
        _ac = True
        logger.info(f"start active set tuning @{i}")
        try:
          dv = constrs_lb.dual()
          idx = dv.argmin()
          # this pops most negative dual variables
          if dv[idx] < 0:
            lb_x.pop(idx)
            lb_y.pop(idx)
            continue
          break
        except Exception as e:
          logger.info(f"finish active set tuning @{i}")
          break
      elif _ac:
        logger.info(f"finish active set tuning @{i}")
        break
      else:
        break

    if _logging:
      logger.info(f"gradient norm: {ndf}")

    if stp <= 1e-6:
      break

    objs = [(i, nabla.obj(x + i / st_line_grids * stp * dp))
            for i in range(1, st_line_grids + 1)]

    i_s, vs = min(objs, key=lambda x: x[-1])
    x = x + i_s / st_line_grids * stp * dp
    if _logging:
      logger.info(f"steps: {i_s}, {vs}, {stp}")
      # update solution
      logger.info(f"obj: {vs}, {vs - _obj}, gap: {(vs - best_obj)/best_obj}")
      logger.info(f"trace deficiency: {n - x.dot(x.T).trace()}")

  return x