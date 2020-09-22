# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /main.py
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 5:19:36 pm
# @description:

from qap_lp.models import *

MSK_DEFAULT = {'mioMaxTime': 60}


def main(instance_name, **kwargs):

  def check_obj_val(x_sol):
    _obj = A0.T.dot(x_sol).dot(B0).dot(x_sol.T).trace()
    print(f'original obj {_obj}')
    return _obj

  # mosek params
  msk_params = {**MSK_DEFAULT, **kwargs}
  # coefficients
  A0, B0 = parse(f'qapdata/{instance_name}.dat')
  A = A0 / np.linalg.norm(A0)
  B = B0 / np.linalg.norm(B0)
  n, m = A.shape
  e = np.ones(shape=n)
  E = np.ones(shape=(n, n))
  ab = np.kron(B.T, A.T)

  param = QAPParam(A, B, n, m, e, E, ab)

  # running tests
  tests = [
      QAPTest('l2_conic_exact', l2_conic_georound, *(param, False),
              **msk_params),
      QAPTest('l2_conic_georound', l2_conic_georound, *(param, True),
              **msk_params),
      QAPTest('l2_naive_exact', l2_naive, *(10, param, True), **msk_params),
      QAPTest('l2_naive_georound', l2_naive, *(10, param, False), **msk_params),
  ]

  objectives = {}
  for t in tests:
    x_sol = t.model(*t.args, **t.kwargs)
    obj = check_obj_val(x_sol)
    objectives[t.name] = obj

  print(objectives)
  return objectives


if __name__ == "__main__":
  main('chr12a')
