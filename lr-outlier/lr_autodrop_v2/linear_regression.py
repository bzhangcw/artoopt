# coding=utf-8
# @author:
# - Meiling Chen, brookechen@sjtu.edu.cn
# select outliers with qp or mip model

from numpy import inf


def qp_gurobipy(x, y0, n, m, N):

    from gurobipy import Model, GRB, quicksum

    # set lamda
    lamda = 1

    # generate model
    model = Model('linear regression_qp')

    # variables
    r = [model.addVar(name='r({})'.format(i), vtype=GRB.CONTINUOUS) for i in range(n)]
    q = [model.addVar(name='q({})'.format(i), vtype=GRB.BINARY) for i in range(n)]
    b = [model.addVar(name='b({})'.format(j), lb=-inf, vtype=GRB.CONTINUOUS) for j in range(m)]

    # objective
    model.setObjective(quicksum(r[i] * q[i] + lamda * (r[i]*r[i] + q[i]*q[i]) for i in range(n)), GRB.MINIMIZE)

    # constraints
    for i in range(n):
        model.addConstr(r[i] >= y0[i] - quicksum(x[j, i] * b[j] for j in range(m)))
        model.addConstr(r[i] >= quicksum(x[j, i] * b[j] for j in range(m)) - y0[i])
    model.addConstr(quicksum(q[i] for i in range(n)) >= N)

    # optimize
    model.optimize()

    # get results
    r_sol = [r[i].x for i in range(n)]
    q_sol = [round(q[i].x) for i in range(n)]
    b_sol = [b[j].x for j in range(m)]

    return r_sol, q_sol, b_sol


def mip_gurobipy(x, y0, n, m, N):

    from gurobipy import Model, GRB, quicksum

    # set big M
    M = 100

    # generate model
    model = Model('linear regression_mip')

    # variables
    r = [model.addVar(name='r({})'.format(i), vtype=GRB.CONTINUOUS) for i in range(n)]
    t = [model.addVar(name='t({})'.format(i), vtype=GRB.CONTINUOUS) for i in range(n)]
    q = [model.addVar(name='q({})'.format(i), vtype=GRB.BINARY) for i in range(n)]
    b = [model.addVar(name='b({})'.format(j), lb=-inf, vtype=GRB.CONTINUOUS) for j in range(m)]

    # objective
    model.setObjective(quicksum(r[i] - t[i] for i in range(n)), GRB.MINIMIZE)

    # constraints
    for i in range(n):
        model.addConstr(r[i] >= y0[i] - quicksum(x[j, i] * b[j] for j in range(m)))
        model.addConstr(r[i] >= quicksum(x[j, i] * b[j] for j in range(m)) - y0[i])
        model.addConstr(t[i] <= M * (1 - q[i]))
        model.addConstr(t[i] <= r[i])
    model.addConstr(quicksum(q[i] for i in range(n)) >= N)

    # optimize
    model.optimize()

    # get results
    r_sol = [r[i].x for i in range(n)]
    q_sol = [round(q[i].x) for i in range(n)]
    b_sol = [b[j].x for j in range(m)]
    t_sol = [t[i].x for i in range(n)]

    return r_sol, q_sol, b_sol


def mip_mip(x, y0, n, m, N, solver_name):

    from mip import Model, xsum, minimize

    # set big M
    M = 100

    # generate model
    model = Model(name='linear regression_mip', solver_name=solver_name)
    model.verbose = 0

    # variables
    r = [model.add_var(name='r({})'.format(i), var_type='C') for i in range(n)]
    t = [model.add_var(name='t({})'.format(i), var_type='C') for i in range(n)]
    q = [model.add_var(name='q({})'.format(i), var_type='B') for i in range(n)]
    b = [model.add_var(name='b({})'.format(j), lb=-inf, var_type='C') for j in range(m)]

    # objective
    model.objective = minimize(xsum(r[i] - t[i] for i in range(n)))

    # constraints
    for i in range(n):
        model += r[i] >= y0[i] - xsum(x[j, i] * b[j] for j in range(m))
        model += r[i] >= xsum(x[j, i] * b[j] for j in range(m)) - y0[i]
        model += t[i] <= M * (1 - q[i])
        model += t[i] <= r[i]
    model += xsum(q[i] for i in range(n)) >= N

    # optimize
    model.optimize()

    # get results
    r_sol = [r[i].x for i in range(n)]
    q_sol = [round(q[i].x) for i in range(n)]
    b_sol = [b[j].x for j in range(m)]
    t_sol = [t[i].x for i in range(n)]

    return r_sol, q_sol, b_sol
