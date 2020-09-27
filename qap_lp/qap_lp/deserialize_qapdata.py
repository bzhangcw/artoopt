# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /deserialize_qapdata.py
# @created: Wednesday, 9th September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Wednesday, 9th September 2020 2:06:20 pm
# @description:

import numpy as np
import re


def parse(path):
  try:
    f = open(path, 'r')
  except:
    raise ValueError("cannot open data file")

  dim = int(f.readline().split('\n')[0])

  data = (float(a) for line in f for a in re.findall('\d+', line))

  arr = np.fromiter(data, dtype=np.float).reshape((2, dim, dim))

  A, B = arr
  return A, B


def parse_sol(path):
  try:
    f = open(path, 'r')
  except:
    raise ValueError("cannot open data file")

  n, obj = map(int, f.readline().strip().split())

  data = (int(a) for line in f for a in re.findall('\d+', line))

  arr = np.fromiter(data, dtype=np.int)

  return n, obj, arr


if __name__ == "__main__":
  import sys

  parse(sys.argv[1])
