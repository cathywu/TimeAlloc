import numpy as np

import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.environ import AbstractModel, RangeSet, Set, Var, Objective, \
    Param, Constraint, summation, Expression

from timealloc.util import fill_from_array, fill_from_2d_array

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
        self.model.dtimeslots = RangeSet(1, self.num_timeslots-1)

        self.model.utilities = Param(self.model.timeslots * self.model.tasks,
                                     initialize=fill_from_2d_array(utilities))

        offdiag = np.zeros((self.num_timeslots, self.num_timeslots))
        offdiag[1:self.num_timeslots, 0:self.num_timeslots-1] = -1
        self.model.offdiag = Param(self.model.timeslots * self.model.timeslots,
                                     initialize=fill_from_2d_array(offdiag))
        self.model.task_duration = Param(self.model.tasks,
                                         initialize=fill_from_array(params[
            'task_duration']))
        self.model.task_chunk_min = Param(self.model.tasks,
                                          initialize=fill_from_array(params[
                                                                         'task_chunk_min']))
        self.model.task_chunk_max = Param(self.model.tasks,
                                          initialize=fill_from_array(params[
                                                                         'task_chunk_max']))
        self._optimized = False

        # useful iterators

        # construct IP
        self._construct_ip()

        # Create a solver
        self.opt = SolverFactory('glpk')

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
            # TODO left off here
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
            return -summation(model.utilities, model.A)

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

    def _constraints_switching_bounds(self):
        """
        Impose bounds on the number of task switches
        TODO(cathywu) impose bounds on the task chunks instead
        """
        def rule(model, i, j):
            """ D[i,j] + (A[i,j) - A[i+1,j]) >= 0 """
            return 0, model.A[i, j] - model.A[i+1, j] + model.D[i, j], None

        self.model.constrain_switching1 = Constraint(self.model.dtimeslots,
                                                     self.model.tasks,
                                                     rule=rule)

        def rule(model, i, j):
            """ D[i,j] - (A[i,j) - A[i+1,j]) >= 0 """
            return 0, -(model.A[i, j] - model.A[i+1, j]) + model.D[i, j], None

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
        # FIXME(cathywu)
        self.integer_program = "blah"
        # variables
        self._variables()
        # objective
        self._objective_cost()
        # self._objective_switching()
        # constraints
        self._constraints_external()
        self._constraints_other()
        self._constraints_nonoverlapping_tasks()
        self._constraints_task_duration()
        self._constraints_switching_bounds()

    def optimize(self):
        # Create a model instance and optimize
        # self.instance = self.model.create_instance("data/calendar.dat")
        self.instance = self.model.create_instance()
        self._results = self.opt.solve(self.instance)
        self._optimized = True

    def display(self):
        self.instance.display()
