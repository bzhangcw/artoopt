# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /test_gradient_proj.p
# @created: Monday, 28th September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 28th September 2020 1:51:15 pm
# @description:

from .models import *
from .conf import *

if __name__ == "__main__":

  instance_name = 'esc16h'
  # instance_name = 'bur26a'
  kwargs = {}
  msk_params = {**MSK_DEFAULT, **kwargs}
  qap_params = {**QAP_DEFAULT, **kwargs}
  # coefficients
  A0, B0 = parse(f'{QAP_INSTANCE}/{instance_name}.dat')

  # parse known solution
  _, best_obj, arr = parse_sol(f'{QAP_SOL}/{instance_name}.sln')

  param = QAPParam(A0, B0, best_obj, arr, **qap_params)

  # unpacking
  A, B, n, m, e, E, ab = \
   param.A, param.B, param.n, param.m, param.e, param.E, param.ab
  # more
  mu = 10
  nabla = QAPDerivativeL2Penalty(param, mu)
  x = x0 = np.ones((n, n)) / n
  nabla.obj(param.xo), nabla.obj(x)

  # try iterations

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
    d0 = nabla.partial_f(x)

    # indices of active lower bound constraints
    lb_x, lb_y = np.where(x <= 1e-4)
    lb_x = lb_x.tolist()
    lb_y = lb_y.tolist()

    if _logging:
      logger.info(f'=====iteration: {i}====')

    # do projection
    while True:

      dp, m, D, constrs_lb, constrs_a, constrs_b = gd_method(
          param,
          d0,
          (lb_x, lb_y),
          logging=True,
      )



      # evaluate norm of the projected gradient
      ndf = np.abs(dp).max()
      # fetch maximum stepsize
      stp = st_method(dp, x, param)
      if ndf <= 1e-6 and stp <= 1e-6:
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
          logger.info(f"cannot find a better active set; KKT satisfied")
          break
      else:
        logger.info(f"finish active set tuning @{i}")
        logger.info(f"cannot find a better active set; KKT satisfied")
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

  print(None)
  print(1)
