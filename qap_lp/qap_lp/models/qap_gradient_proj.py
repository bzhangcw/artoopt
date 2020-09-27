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

  model.constraint(expr.sum(D, 0), dom.equalsTo(0))
  model.constraint(expr.sum(D, 1), dom.equalsTo(0))
  model.constraint(expr.vstack(v, m), dom.inQCone())
  constrs_lb = model.constraint(D.pick(*lb_indices), dom.equalsTo(0))
  # constrs_ub = model.constraint(D.pick(*ub_indices), dom.lessThan(0))

  # set params
  userCallback = set_mosek_model_params(model, **kwargs)
  model.setLogHandler(None)
  model.solve()

  model.flushSolutions()
  D_sol = D.level().reshape(A.shape)
  return D_sol, model, D, constrs_lb


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


def run_gradient_projection(x, mu, param: QAPParam, nabla: QAPDerivative,
                            **kwargs):
  # unpacking solver parameters
  max_iter = kwargs.get('max_iteration', 1000)
  gd_method = kwargs.get('gd_method', msk_pd_on_dc)
  st_method = kwargs.get('st_method', msk_st)
  st_line_grids = kwargs.get('st_line_grids', 10)

  # unpacking params
  n = param.n
  xo = param.xo
  best_obj = nabla.obj(xo)

  # start iterations
  for i in range(max_iter):
    logger.info(f'=====iteration: {i}====')
    _obj = nabla.obj(x)
    d0 = nabla.partial_f(x)

    # indices of active lower bound constraints
    lb_x, lb_y = np.where(x <= 1e-4)
    lb_x = lb_x.tolist()
    lb_y = lb_y.tolist()

    # do projection
    dp, m, D, constrs_lb = gd_method(
        param,
        d0,
        (lb_x, lb_y),
    )

    # evaluate norm of the projected gradient
    ndf = np.abs(dp).max()

    # fetch maximum stepsize
    stp = st_method(dp, x, param)

    logger.info(f"gradient norm: {ndf}")

    if stp <= 1e-6:
      #
      #         try:
      #             idx = constrs_lb.dual().argmax()
      #             lb_x.pop(idx)
      #             lb_y.pop(idx)
      #         except Exception as e:
      #             logging.exception(e)
      #             lb_x, lb_y = np.where(x <= 1e-5)
      #             lb_x = lb_x.tolist()
      #             lb_y = lb_y.tolist()
      #         logger.info(f"change active set @{i}")
      #         dp, m, D, constrs_lb = pd_on_dc(
      #             param, d0,
      #             (lb_x, lb_y)
      #         )
      break

    objs = [(i, nabla.obj(x + i / st_line_grids * stp * dp))
            for i in range(1, st_line_grids + 1)]
    i_s, vs = min(objs, key=lambda x: x[-1])
    logger.info(f"steps: {i_s}, {vs}, {stp}")
    # update solution
    x = x + i_s / st_line_grids * stp * dp
    logger.info(f"obj: {vs, vs - _obj}, gap: {(vs - best_obj)/best_obj}")
    logger.info(f"trace deficiency: {n - x.dot(x.T).trace()}")

  return x