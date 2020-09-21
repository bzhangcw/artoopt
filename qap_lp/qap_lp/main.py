# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /main.py
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 5:19:36 pm
# @description:

from qap_lp.models import *


def check_obj_val(x_sol):
  _obj = A0.T.dot(x_sol).dot(B0).dot(x_sol.T).trace()
  print(f'original obj {_obj}')
  return _obj


if __name__ == "__main__":
  # parameters
  A0, B0 = parse('qapdata/nug12.dat')
  A = A0 / np.linalg.norm(A0)
  B = B0 / np.linalg.norm(B0)
  n, m = A.shape
  e = np.ones(shape=n)
  E = np.ones(shape=(n, n))
  ab = np.kron(B.T, A.T)

  param = QAPParam(A, B, n, m, e, E, ab)

  tests = [
      QAPTest('co_l2_exact', co_gm, *(param, False, 1000)),
      QAPTest('co_l2_georounding', co_gm, *(param, True, 60)),
      QAPTest('co_l2_naive', l2_naive, *(10, param, True))
  ]
  # # x_co = co_gm(param, False, 60)
  # # check_obj_val(x_co)
  # # import pickle as pk
  # # pk.dump(x_co, open("x_co.pk", 'wb'))
  # # print(x_co)
  # x_co_gm = co_gm(param, True, 60)
  # check_obj_val(x_co_gm)
  # import pickle as pk
  # pk.dump(x_co_gm, open("x_co_gm.pk", 'wb'))
  # print(x_co_gm)
  # # x_l2_naive = l2_naive(10, param, rd=True)
  # # check_obj_val(x_l2_naive)
  # # pk.dump(x_l2_naive, open("x_l2_naive.pk", 'wb'))
  # # print(x_l2_naive)
  objectives = {}
  for t in tests:
    x_sol = t.model(*t.args, **t.kwargs)
    obj = check_obj_val(x_sol)
    objectives[t.name] = obj

  print(objectives)
