import logging

# solver sanity check
BOOL_HAS_COPT = True
BOOL_HAS_GRB = True

logger = logging.getLogger("sfhub.util")
try:
    import coptpy
except ImportError as e:
    logger.warning("Cannot find COPT & coptpy")
    # logger.exception(e)
    BOOL_HAS_COPT = False
try:
    import gurobipy
except ImportError as e:
    logger.warning("Cannot find GUROBI & gurobipy")
    BOOL_HAS_GRB = False

try:
    import mosek.fusion as mf
    expr = mf.Expr
    dom = mf.Domain
    mat = mf.Matrix
except ImportError as e:
    logger.warning('no mosek detected')


class ModelWrapper(object):
    """
    a wrapper class to work with different solvers
    """

    def __init__(self, model=None, object_map=None, solver_name="", name="model", *args, **kwargs):
        _solver_name = solver_name.upper()
        if model:
            self.model = model
        else:
            if _solver_name == 'COPT':
                if BOOL_HAS_COPT:
                    envr = coptpy.Envr()
                    self.model = envr.createModel(name=name)
                else:
                    raise ValueError("Cannot find COPT!")
            elif _solver_name == 'GUROBI':
                if BOOL_HAS_GRB:
                    self.model = gurobipy.Model(name)
                else:
                    logger.warning('Cannot find GUROBI, pls install the API properly')
            else:
                logger.info('Unknown solver, fallback to COPT')
                try:
                    envr = coptpy.Envr()
                    self.model = envr.createModel(name=name)
                except Exception as e:
                    logger.error("Cannot find COPT!")
                    raise e
        self.obj_map = object_map
        self._objective_value = None
        self.is_copt, self.is_grb = False, False
        if BOOL_HAS_COPT:
            self.is_copt = self.model.__class__ == coptpy.Model
        if BOOL_HAS_GRB:
            self.is_grb = self.model.__class__ == gurobipy.Model

        if not (self.is_grb or self.is_copt):
            raise ValueError("unsupported, neither COPT nor GUROBI")

        # wrapper constants
        if self.is_copt:
            self.INTEGER = coptpy.COPT.INTEGER
            self.BINARY = coptpy.COPT.BINARY
            self.CONTINUOUS = coptpy.COPT.CONTINUOUS
            self.INF = coptpy.COPT.INFINITY
            self.MINIMIZE = coptpy.COPT.MINIMIZE
            self.xsum = self.quicksum = coptpy.quicksum
        elif self.is_grb:
            self.INTEGER = gurobipy.GRB.INTEGER
            self.BINARY = gurobipy.GRB.BINARY
            self.CONTINUOUS = gurobipy.GRB.CONTINUOUS
            self.INF = gurobipy.GRB.INFINITY
            self.MINIMIZE = gurobipy.GRB.MINIMIZE
            self.xsum = self.quicksum = gurobipy.quicksum
        else:
            raise ValueError("solver name unknown")

        self.set_properties(**kwargs)

    def optimize(self, **kwargs):
        time_limit = kwargs.get("max_seconds", 1000)
        max_solutions = kwargs.get("max_solutions", 1000)
        if self.is_copt:
            return self.model.solve()
        elif self.is_grb:
            return self.model.optimize()
        raise TypeError("unknown model class, not in [copt, grb]")

    @property
    def objective_value(self):
        if self.is_copt:
            value = self.model.objval
        elif self.is_grb:
            value = self.model.objVal
        else:
            value = 0.0
        self._objective_value = value
        return self._objective_value

    def isfeasible(self):
        """
        Not safe, only after optimize call!
        :return:
        """
        if self.is_copt:
            return self.model.status != coptpy.COPT.INFEASIBLE
        if self.is_grb:
            return self.model.status != gurobipy.GRB.INFEASIBLE
        return False

    def set_properties(self, **kwargs):
        verbose, max_mip_gap = kwargs.get('verbose', 1), kwargs.get('maxMipGap', 0.00)
        if self.is_copt:
            self.model.setParam('Logging', verbose)
            self.model.setParam('RelGap', max_mip_gap)
        elif self.is_grb:
            self.model.setParam('OutputFlag', verbose)
            self.model.setParam('MIPGap', max_mip_gap)
        else:
            raise ValueError("solver name unknown")
