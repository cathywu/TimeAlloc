import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.environ import AbstractModel, RangeSet, Set, Var, Objective, \
    Param, Constraint, summation

from timealloc.util import fill_from_array

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

        self.model.utilities = Param(self.model.timeslots * self.model.tasks,
                                     initialize=fill_from_array(utilities),
                                     default=1)
        self._optimized = False

        # useful iterators

        # construct IP
        self._construct_ip()

        # Create a solver
        self.opt = SolverFactory('glpk')

    def _variables(self):
        self.model.A = Var(self.model.timeslots * self.model.tasks,
                           domain=pe.Boolean)

    def _objective_switching(self):
        """
        Reward task-specific amounts of task switching
        """
        pass

    def _objective_cost(self):
        """ Objective function to minimize """
        def obj_expression(model):
            return -summation(model.utilities, model.A)

        self.model.OBJ = Objective(rule=obj_expression)

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
        No multi-tasking! No events should take place at the same time.
        Perhaps a future version can account for light-weight multi-tasking,
        like commuting + reading.
        """
        def rule(model, j):
            return 0, sum(model.A[i, j] for i in model.timeslots), None

        self.model.constrain_task_duration = Constraint(self.model.tasks,
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
        # constraints
        self._constraints_external()
        self._constraints_other()
        self._constraints_nonoverlapping_tasks()
        self._constraints_task_duration()

    def optimize(self):
        # Create a model instance and optimize
        # self.instance = self.model.create_instance("data/calendar.dat")
        self.instance = self.model.create_instance()
        self._results = self.opt.solve(self.instance)
        self._optimized = True

    def display(self):
        self.instance.display()
