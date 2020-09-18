# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /qap_georounding.py
# @created: Friday, 18th September 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Friday, 18th September 2020 11:26:25 am
# @description:
# ===
# geometric rounding
# ===
def geo_round(x, u, domain=None):
  x_selected = set()
  x_int = []
  _domain = domain if domain else set(range(x.shape[1]))
  for x_slice, x_sort in zip(x, np.argsort(u / x, 1)):
    if abs(x_slice.max() - 1) < 1e-3:
      x_int.append(x_slice.tolist())
      x_selected.add(np.argmax(x_slice))
      continue
    for j in x_sort:
      if j in _domain and (j not in x_selected):
        break
    x_vec = [0] * x.shape[1]
    x_vec[j] = 1
    x_selected.add(j)
    x_int.append(x_vec)
  x_int = np.array(x_int)
  return x_int, x_selected


def extract_sol_rounding(x, A, B, max_iterations=1e3):

  def get_sol_and_obj(x, u):
    x_int, x_selected = geo_round(x, u)
    return x_int, A.transpose().dot(x_int).dot(B).dot(x_int.transpose()).trace()

  rs = np.random.mtrand.RandomState()
  rs.seed(1)

  def sampling():
    for i in range(int(max_iterations)):
      a = rs.exponential(1, size=x.shape[1])
      u = a / a.sum()
      yield get_sol_and_obj(x, u)

  x_val_array = np.array(list(sampling()))

  # print out sampling status
  x, _ = min(x_val_array, key=lambda x: x[-1])
  obj_vals = x_val_array[:, -1]
  #
  print(
      f"sampling results: min:{obj_vals.min()}, max:{obj_vals.max()}, avg:{obj_vals.mean()}"
  )

  return x, _
