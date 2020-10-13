# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /main.py
# @created: Monday, 21st September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 21st September 2020 5:19:36 pm
# @description:

import argparse
import json
import sys
import pickle as pk

from .models import *
from .conf import *


def write_sol(instance_name, test_name, x_sol):
  x = x_sol.round(3)
  with open(f"{RESULT_DIR}/{instance_name}_{test_name}.pk", 'wb') as f:
    pk.dump(x, f)

  with open(f"{RESULT_DIR}/{instance_name}_{test_name}", 'w') as f1:
    y = (x > 0).nonzero()
    list_sol = list(zip(*y))
    for i, j in list_sol:
      f1.write(f"{i}, {j}\n")

  return 1


def main_single(instance_name, **kwargs):

  # mosek params
  msk_params = {**MSK_DEFAULT, **kwargs}
  qap_params = {**QAP_DEFAULT, **kwargs}

  # coefficients
  A0, B0 = parse(f'{QAP_INSTANCE}/{instance_name}.dat')

  # parse known solution
  _, best_obj, arr = parse_sol(f'{QAP_SOL}/{instance_name}.sln')

  param = QAPParam(A0, B0, best_obj, arr, **qap_params)

  logging.info(f"problem @{instance_name} parsing finished")
  # running tests
  tests = [
      QAPTest('l2_exact_penalty_gradient_proj', l2_exact_penalty_gradient_proj,
              *(param,), **qap_params),
      # QAPTest('l2_conic_exact', l2_conic_georound, *(param, False), **msk_params),
      # QAPTest('l2_conic_georound', l2_conic_georound, *(param, True),**msk_params),
      # QAPTest('l2_naive_exact', l2_naive, *(10, param, True), **msk_params),
      # QAPTest('l2_naive_georound', l2_naive, *(10, param, False), **msk_params),
  ]

  objectives = {
      'best': {
          'instance': instance_name,
          'value': best_obj,
          'rel_gap': 0,
          'trace_res': 0
      }
  }
  for t in tests:
    x_sol = t.run()
    write_sol(instance_name, t.name, x_sol)
    obj = check_obj_val(param, x_sol)
    objectives[t.name] = {
        'instance': instance_name,
        'value': obj,
        'rel_gap': (obj - best_obj) / best_obj,
        'trace_res': param.n - x_sol.dot(x_sol.T).trace(),
        'runtime': t.end - t.start
    }

  import json
  format_obj_str = json.dumps(objectives, indent=2)
  logging.info(f"=== objectives: ===\n{format_obj_str}")
  # dump the solution quality file
  json.dump(objectives, open(f"{RESULT_DIR}/{instance_name}.json", 'w'))

  return objectives


def main(**kwargs):
  instance = kwargs.get('instance')
  failed = []
  if instance:
    main_single(instance, **kwargs)
  else:
    files = os.listdir(QAP_INSTANCE)
    for f in files:
      try:
        ins, suffix = f.split('.')
        objectives = main_single(ins, **kwargs)
        with open(f"{RESULT_DIR}/{ins}.json", 'w') as f_out:
          json.dump(objectives, fp=f_out)
      except Exception as e:
        logging.error(f"failed @{ins}")
        failed.append(ins)
  logging.error(f"failed instances: \n{failed}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--instance', default=None)
  parser.add_argument('--logging_interval', type=int, default=50)
  parser.add_argument('--max_iteration', type=int, default=1000)
  kwargs = parser.parse_args()
  main(**vars(kwargs))
