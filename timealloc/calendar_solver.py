import numpy as np

import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.environ import AbstractModel, RangeSet, Set, Var, Objective, \
    Param, Constraint, summation, Expression

from timealloc.util import fill_from_array, fill_from_2d_array
import timealloc.util as util

class CalendarSolver:
    """
    Creates an integer program (IP) based on scheduling constraints and
    preferences.
    """

    def __init__(self, utilities, params):
        self.model = AbstractModel()

        # read parameters
        # self.num_tasks = Param(initialize=params['num_tasks'], default=5)
        # self.num_timeslots = Param(initialize=params['num_timeslots'],
        #                            default=10)
        self.num_tasks = params['num_tasks']
        self.num_timeslots = params['num_timeslots']
        self.model.tasks = RangeSet(1, self.num_tasks)
        self.model.timeslots = RangeSet(1, self.num_timeslots)
        self.model.dtimeslots = RangeSet(1, self.num_timeslots - 1)

        self.model.utilities = Param(self.model.timeslots * self.model.tasks,
                                     initialize=fill_from_2d_array(utilities))

        offdiag = np.zeros((self.num_timeslots, self.num_timeslots))
        offdiag[1:self.num_timeslots, 0:self.num_timeslots - 1] = -1
        self.model.offdiag = Param(self.model.timeslots * self.model.timeslots,
                                   initialize=fill_from_2d_array(offdiag))
        self.model.task_duration = Param(self.model.tasks,
                                         initialize=fill_from_array(
                                             params['task_duration']))
        self.model.task_chunk_min = Param(self.model.tasks,
                                          initialize=fill_from_array(
                                              params['task_chunk_min']))
        self.model.task_chunk_max = Param(self.model.tasks,
                                          initialize=fill_from_array(
                                              params['task_chunk_max']))
        self._optimized = False

        # useful iterators

        # construct IP
        self._construct_ip()

        # Create a solver
        # self.opt = SolverFactory('glpk')
        # self.opt.options['tmlim'] = 1000
        # self.opt = SolverFactory('ipopt')
        # self.opt.options['max_iter'] = 10000
        self.opt = SolverFactory('cbc')

    def _variables(self):
        # allocation A
        self.model.A = Var(self.model.timeslots * self.model.tasks,
                           domain=pe.Boolean)
        # delta D
        self.model.D = Var(self.model.dtimeslots * self.model.tasks,
                           domain=pe.Integers)

    def _objective_switching(self):
        """
        Reward task-specific amounts of task switching
        """
        def obj_expression(model):
            total = 0
            for j in range(1, self.num_tasks+1):
                for i in range(1, self.num_timeslots):
                    total += abs(model.A[i, j] - model.A[i+1, j])
            return total

        # TODO objective with multiple parts
        self.model.exp_switching = Expression(rule=obj_expression)

    def _objective_cost(self):
        """ Objective function to minimize """
        def obj_expression(model):
            return -(summation(model.utilities, model.A))
            # TODO(cathywu) this objective may slow things down, not solvable
            # via glpk
            # return -(summation(model.utilities, model.A) + 6 * pe.summation(
            #     model.C6) + 4 * pe.summation(model.C4) + 3 * pe.summation(
            #     model.C3) + pe.summation(model.C2))

        # self.model.exp_cost = Expression(rule=obj_expression)
        # self.model.obj_cost = Objective(rule=self.model.exp_cost)
        self.model.obj_cost = Objective(rule=obj_expression)

    def _constraints_external(self):
        """
        Hard constraints from external calendar (e.g. pre-scheduled meetings).
        """
        pass

    def _constraints_other(self):
        """ Other constraints, user imposed, like keeping Friday night free """
        pass

    def _constraints_nonoverlapping_tasks(self):
        """
        No multi-tasking! No events should take place at the same time.
        Perhaps a future version can account for light-weight multi-tasking,
        like commuting + reading.
        """
        def rule(model, i):
            return 0, sum(model.A[i, j] for j in model.tasks), 1

        self.model.constrain_nonoverlapping = Constraint(self.model.timeslots,
                                                         rule=rule)

    def _constraints_task_duration(self):
        """
        Each task should stay within task-specific allocation bounds
        """
        def rule(model, j):
            task_j_total = sum(model.A[i, j] for i in model.timeslots)
            return 0, task_j_total, model.task_duration[j]

        self.model.constrain_task_duration = Constraint(self.model.tasks,
                                                        rule=rule)

    def _constraints_chunking(self):
        """
        Ensures that there are no tasks allocated for only 1 slot
        """
        chunk_tiny1 = np.array([-1, 1, -1])

        L0, b0 = util.linop_from_1d_filter(chunk_tiny1, self.num_timeslots,
                                           offset=1)
        L1, b1 = util.linop_from_1d_filter(1 - chunk_tiny1, self.num_timeslots,
                                           offset=1)

        offset = 1
        c3len = self.num_timeslots - chunk_tiny1.size + 1 + offset * 2
        self.model.c3timeslots = RangeSet(1, c3len)
        self.model.cL0 = Param(self.model.c3timeslots * self.model.timeslots,
                               initialize=fill_from_2d_array(L0))
        self.model.cL1 = Param(self.model.c3timeslots * self.model.timeslots,
                               initialize=fill_from_2d_array(L1))
        self.model.C1 = Var(self.model.c3timeslots * self.model.tasks,
                            domain=pe.Reals)

        def rule(model, i, j):
            return None, model.C1[i, j], 0

        self.model.constrain_chunking0 = Constraint(self.model.c3timeslots,
                                                    self.model.tasks, rule=rule)

        def rule(model, i, j):
            total = sum(
                model.cL0[i, k] * model.A[k, j] for k in model.timeslots)
            return 0, model.C1[i, j] - total, None
            # return - 1, model.C3[i, j] - total / chunk3.size, 0

        self.model.constrain_chunking = Constraint(self.model.c3timeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking2(self):
        """
        Ensures that there are no tasks allocated for only 2 slots
        :return:
        """
        chunk_tiny1 = np.array([-1, 1, 1, -1])

        L0, b0 = util.linop_from_1d_filter(chunk_tiny1, self.num_timeslots,
                                           offset=1)

        offset = 1
        c2len = self.num_timeslots - chunk_tiny1.size + 1 + offset * 2
        self.model.c2timeslots = RangeSet(1, c2len)
        self.model.cL20 = Param(self.model.c2timeslots * self.model.timeslots,
                                initialize=fill_from_2d_array(L0))
        self.model.C2 = Var(self.model.c2timeslots * self.model.tasks,
                            domain=pe.Reals)

        def rule(model, i, j):
            return None, model.C2[i, j], 1

        self.model.constrain_chunking20 = Constraint(self.model.c2timeslots,
                                                     self.model.tasks,
                                                     rule=rule)

        def rule(model, i, j):
            total = sum(
                model.cL20[i, k] * model.A[k, j] for k in model.timeslots)
            return 0, model.C2[i, j] - total, None

        self.model.constrain_chunking2 = Constraint(self.model.c2timeslots,
                                                    self.model.tasks, rule=rule)

    def _constraints_chunking_max(self):
        """
        Ensures there are no tasks allocated beyond a maximum chunk length

        FIXME(cathywu) this seems to slow down solving significantly.
        """
        chunk_tiny1 = np.array([1, 1, 1, 1, 1, 1])
        # chunk_max = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        # chunk_tiny1 = np.array([-1, 1, 1, -1])

        offset = 0
        L0, b0 = util.linop_from_1d_filter(chunk_tiny1, self.num_timeslots,
                                           offset=offset)

        cmaxlen = self.num_timeslots - chunk_tiny1.size + 1 + offset * 2
        self.model.cmaxtimeslots = RangeSet(1, cmaxlen)
        self.model.cLmax0 = Param(
            self.model.cmaxtimeslots * self.model.timeslots,
            initialize=fill_from_2d_array(L0))
        self.model.Cmax = Var(self.model.cmaxtimeslots * self.model.tasks,
                              domain=pe.Reals)

        def rule(model, i, j):
            return None, model.Cmax[i, j], chunk_tiny1.size - 1

        self.model.constrain_chunking_max0 = Constraint(
            self.model.cmaxtimeslots, self.model.tasks, rule=rule)

        def rule(model, i, j):
            total = sum(
                model.cLmax0[i, k] * model.A[k, j] for k in model.timeslots)
            return 0, model.Cmax[i, j] - total, None

        self.model.constrain_chunking_max = Constraint(self.model.cmaxtimeslots,
                                                       self.model.tasks,
                                                       rule=rule)

    def _constraints_switching_bounds(self):
        """
        Impose bounds on the number of task switches
        TODO(cathywu) impose bounds on the task chunks instead
        """

        def rule(model, i, j):
            """ D[i,j] + (A[i,j) - A[i+1,j]) >= 0 """
            return 0, model.A[i, j] - model.A[i + 1, j] + model.D[i, j], None

        self.model.constrain_switching1 = Constraint(self.model.dtimeslots,
                                                     self.model.tasks,
                                                     rule=rule)

        def rule(model, i, j):
            """ D[i,j] - (A[i,j) - A[i+1,j]) >= 0 """
            return 0, -(model.A[i, j] - model.A[i + 1, j]) + model.D[i, j], None

        self.model.constrain_switching2 = Constraint(self.model.dtimeslots,
                                                     self.model.tasks,
                                                     rule=rule)

        def rule(model, i, j):
            """ 0 <= D[i,j] <= 1 """
            return 0, model.D[i, j], 1

        self.model.constrain_switching3 = Constraint(self.model.dtimeslots,
                                                     self.model.tasks,
                                                     rule=rule)

        def rule(model, j):
            switches = sum(model.D[i, j] for i in model.dtimeslots) / 2
            return model.task_duration[j] / model.task_chunk_max[j], switches, \
                   model.task_duration[j] / model.task_chunk_min[j]

        self.model.constrain_switching4 = Constraint(self.model.tasks,
                                                     rule=rule)

    def _construct_ip(self):
        """
        Aggregates IP construction
        """
        # name the problem
        self.integer_program = "CalenderSolver"
        # variables
        self._variables()
        # constraints
        self._constraints_external()
        self._constraints_other()
        self._constraints_nonoverlapping_tasks()
        self._constraints_task_duration()
        self._constraints_switching_bounds()
        self._constraints_chunking()
        self._constraints_chunking2()
        self._constraints_chunking_max()
        # objective
        self._objective_cost()
        # self._objective_switching()

    def optimize(self):
        # Create a model instance and optimize
        # self.instance = self.model.create_instance("data/calendar.dat")
        self.instance = self.model.create_instance()
        self._results = self.opt.solve(self.instance)
        self._optimized = True

    def display(self):
        self.instance.display()
