import operator

import numpy as np

from bokeh.palettes import d3
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, SingleIntervalTicker, LinearAxis, \
    LabelSet, Range1d
import pyomo.environ as pe
from pyomo.environ import AbstractModel, RangeSet, Set, Var, Objective, Param, \
    Constraint, summation, Expression
from pyomo.opt import SolverFactory

from timealloc.util import fill_from_array, fill_from_2d_array
import timealloc.util as util
import timealloc.util_time as tutil

# For avoiding rounding issues
EPS = 1e-2  # epsilon

# Time limit for solver (wallclock)
TIMELIMIT = 2e2  # 3600, 1e3, 2e2, 50

# granularity (in hours) for contiguity variables (larger --> easier problem)
CONT_STRIDE = 12

# slack for contiguity variables (larger --> easier problem)
SLACK = 5


class CalendarSolver:
    """
    Creates an integer program (IP) based on scheduling constraints and
    preferences.
    """

    def __init__(self, utilities, params):
        self.model = AbstractModel()
        self._optimized = False

        self.slack_cont = SLACK

        # read parameters
        # self.num_tasks = Param(initialize=params['num_tasks'], default=5)
        # self.num_timeslots = Param(initialize=params['num_timeslots'],
        #                            default=10)
        self.num_tasks = params['num_tasks']
        self.num_timeslots = params['num_timeslots']
        self.valid = params['task_valid']
        if 'task_names' in params:
            self.task_names = params['task_names']
        else:
            self.task_names = ["" for _ in range(self.num_tasks)]
        self.task_duration = params['task_duration']
        self.task_chunk_min = params['task_chunk_min']
        self.task_chunk_max = params['task_chunk_max']
        self.valid = params['task_valid']

        # Index sets for iteration
        self.model.tasks = RangeSet(0, self.num_tasks - 1)
        self.model.timeslots = RangeSet(0, self.num_timeslots - 1)
        self.model.dtimeslots = RangeSet(0, self.num_timeslots - 2)

        # Fill pyomo Params from user params
        self.utilities = utilities

        # construct IP
        self._construct_ip()

        # Create a solver
        # self.opt = SolverFactory('glpk')
        # self.opt = SolverFactory('ipopt')
        self.opt = SolverFactory('cbc')
        # self.opt.options['tmlim'] = 1000  # glpk
        # self.opt.options['max_iter'] = 10000  # ipopt
        # self.opt.options['timelimit'] = 5

    def _variables(self):
        """
        Primary variables are defined here. Intermediate variables may be
        defined directly in constraints.

        Convention: variables are capitalized, i.e. model.A, and not model.a
        """
        # allocation A
        self.model.A = Var(self.model.timeslots * self.model.tasks,
                           domain=pe.Boolean)
        self.model.A_total = Var(domain=pe.Reals)
        # delta D
        # TODO(cathywu) consider whether this / switching constraints are needed
        self.model.D = Var(self.model.dtimeslots * self.model.tasks,
                           domain=pe.Integers)

    def _objective_switching(self):
        """
        Reward task-specific amounts of task switching
        """

        def obj_expression(model):
            total = 0
            for j in range(1, self.num_tasks + 1):
                for i in range(1, self.num_timeslots):
                    total += abs(model.A[i, j] - model.A[i + 1, j])
            return total

        # TODO objective with multiple parts
        self.model.exp_switching = Expression(rule=obj_expression)

    def _objective_cost(self):
        """ Objective function to minimize """

        def obj_expression(model):
            return -(model.A_total + model.CTu_total + model.CTl_total)
            # return -(model.A_total)
            # model.A_total + model.CTu / self.slack_cont + model.CTl /
            # self.slack_cont)
            # return -(summation(self.utilities, model.A) + summation(
            #     model.CTu) / self.slack_cont + summation(
            #     model.CTl) / self.slack_cont)

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

    def _constraints_task_valid(self):
        """
        User-defined time constraints on tasks
        """

        def rule(model, i, j):
            return 0, model.A[i, j] * (1-self.valid[i, j]), 0

        self.model.constrain_valid = Constraint(self.model.timeslots,
                                                self.model.tasks, rule=rule)

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
            return 0, task_j_total, self.task_duration[j]

        self.model.constrain_task_duration = Constraint(self.model.tasks,
                                                        rule=rule)

    def _constraints_utility(self):
        """
        Each task should stay within task-specific allocation bounds
        """

        def rule(model):
            total = summation(self.utilities, model.A)
            return model.A_total == total

        self.model.constrain_A_total = Constraint(rule=rule)

    def _constraints_task_contiguity_linear(self):
        """
        Encourage the chunks of a tasks to be scheduled close to one another,
        i.e. reward shorter "elapsed" times
        """
        triu = np.triu(np.ones(self.num_timeslots))
        tril = np.tril(np.ones(self.num_timeslots))

        self.model.CTu = Var(domain=pe.Integers)
        self.model.CTl = Var(domain=pe.Integers)

        def rule(model):
            """
            This rule is used to encourage early completion (in terms of
            allocation) of a task.

            More precisely:
            CTu[i,j] = whether task j is UNASSIGNED between slot i and the end

            Maximizing sum_i CTu[i,j] encourages early task completion.
            Maximizing sum_i CTu[i,j]+CTl[i,j] encourages contiguous scheduling.
            """
            total = 0
            ind = model.timeslots
            for i in model.timeslots:
                for j in model.tasks:
                    den = self.num_timeslots - i
                    total += sum(triu[i, k] * (1-model.A[k, j]) for k in
                                 ind) / den
            # total = sum(model.cTu[i, k] * (1-model.A[k, j]) for k in ind) /
            #  den
            return -1 + EPS, model.CTu - total, EPS + self.slack_cont

        self.model.constrain_contiguity_u = Constraint(rule=rule)

        def rule(model):
            """
            This rule is used to encourage late start (in terms of
            allocation) of a task.

            More precisely:
            CTl[i,j] = whether task j is UNASSIGNED between slot 0 and slot i

            Maximizing sum_i CTl[i,j] encourages late starting.
            Maximizing sum_i CTu[i,j]+CTl[i,j] encourages contiguous scheduling.
            """
            total = 0
            ind = model.timeslots
            for i in model.timeslots:
                for j in model.tasks:
                    den = i + 1
                    total = sum(tril[i, k] * (1-model.A[k, j]) for k in
                                ind) / den
            # total = sum(model.cTl[i, k] * (1-model.A[k, j]) for k in ind) /
            #  den
            return -1 + EPS, model.CTl - total, EPS + self.slack_cont

        self.model.constrain_contiguity_l = Constraint(rule=rule)

    def _constraints_task_contiguity(self):
        """
        Encourage the chunks of a tasks to be scheduled close to one another,
        i.e. reward shorter "elapsed" times
        """
        # triu = np.triu(np.ones(self.num_timeslots))
        # tril = np.tril(np.ones(self.num_timeslots))
        incr = CONT_STRIDE * tutil.SLOTS_PER_HOUR  # 1 would give original result
        triu = util.triu(self.num_timeslots, incr=incr)
        tril = util.tril(self.num_timeslots, incr=incr)
        cont_slots = self.num_timeslots/incr-1

        self.model.contslots = RangeSet(0, cont_slots - 1)
        self.model.CTu = Var(self.model.contslots * self.model.tasks,
                             domain=pe.Integers)
        self.model.CTl = Var(self.model.contslots * self.model.tasks,
                             domain=pe.Integers)
        self.model.CTu_total = Var(domain=pe.Reals)
        self.model.CTl_total = Var(domain=pe.Reals)

        def rule(model, i, j):
            """
            This rule is used to encourage early completion (in terms of
            allocation) of a task.

            More precisely:
            CTu[i,j] = whether task j is UNASSIGNED between slot i and the end

            Maximizing sum_i CTu[i,j] encourages early task completion.
            Maximizing sum_i CTu[i,j]+CTl[i,j] encourages contiguous scheduling.
            """
            den = sum(triu[i, :])
            ind = model.timeslots
            total = sum(triu[i, k] * (1-model.A[k, j]) for k in ind) / den
            return -1 + EPS, model.CTu[i, j] - total, EPS + self.slack_cont

        self.model.constrain_contiguity_u = Constraint(self.model.contslots,
                                                       self.model.tasks,
                                                       rule=rule)

        def rule(model, i, j):
            """
            This rule is used to encourage late start (in terms of
            allocation) of a task.

            More precisely:
            CTl[i,j] = whether task j is UNASSIGNED between slot 0 and slot i

            Maximizing sum_i CTl[i,j] encourages late starting.
            Maximizing sum_i CTu[i,j]+CTl[i,j] encourages contiguous scheduling.
            """
            den = sum(tril[i, :])
            ind = model.timeslots
            total = sum(tril[i, k] * (1-model.A[k, j]) for k in ind) / den
            return -1 + EPS, model.CTl[i, j] - total, EPS + self.slack_cont

        self.model.constrain_contiguity_l = Constraint(self.model.contslots,
                                                       self.model.tasks,
                                                       rule=rule)

        def rule(model):
            den = self.num_tasks * cont_slots * (self.slack_cont + 1)
            num = 0.25
            total = summation(model.CTu) / den * num
            return model.CTu_total == total

        self.model.constrain_contiguity_ut = Constraint(rule=rule)

        def rule(model):
            den = self.num_tasks * cont_slots * (self.slack_cont + 1)
            num = 0.25
            total = summation(model.CTl) / den * num
            return model.CTl_total == total

        self.model.constrain_contiguity_lt = Constraint(rule=rule)

    def _constraints_chunking1(self):
        """
        Ensures that there are no tasks allocated for only 1 slot
        """
        chunk_len = 1
        offset = 1
        filter = np.ones(chunk_len + offset * 2)
        filter[0:offset] = -1
        filter[-offset:] = -1
        # filter = np.array([-1, 1, -1])
        L, b = util.linop_from_1d_filter(filter, self.num_timeslots,
                                         offset=offset)
        c_len = self.num_timeslots - filter.size + 1 + offset * 2

        self.model.cmin1_timeslots = RangeSet(0, c_len - 1)
        self.model.C1 = Var(self.model.cmin1_timeslots * self.model.tasks,
                            domain=pe.Reals)
        var_name = 'C1'

        def rule(model, i, j):
            """
            C[i, j]==1 means that the pattern is matched, anything less is okay
            """
            C = operator.attrgetter(var_name)(model)[i, j]
            if self.task_chunk_min[j] <= chunk_len:
                return Constraint.Feasible
            return None, C, chunk_len - 1

        self.model.constrain_chunk10 = Constraint(self.model.cmin1_timeslots,
                                                  self.model.tasks, rule=rule)

        def rule(model, i, j):
            C = operator.attrgetter(var_name)(model)[i, j]
            if self.task_chunk_min[j] <= chunk_len:
                return Constraint.Feasible
            total = sum(L[i, k] * model.A[k, j] for k in model.timeslots)
            return 0, C - total, None

        self.model.constrain_chunk11 = Constraint(self.model.cmin1_timeslots,
                                                  self.model.tasks, rule=rule)

    def _constraints_chunking2(self):
        """
        Ensures that there are no tasks allocated for only 2 slots
        """
        chunk_len = 2
        offset = 1
        filter = np.ones(chunk_len + offset * 2)
        filter[0:offset] = -1
        filter[-offset:] = -1
        # filter = np.array([-1, 1, 1, -1])
        L, b = util.linop_from_1d_filter(filter, self.num_timeslots,
                                         offset=offset)
        c_len = self.num_timeslots - filter.size + 1 + offset * 2

        self.model.c2timeslots = RangeSet(0, c_len - 1)
        self.model.C2 = Var(self.model.c2timeslots * self.model.tasks,
                            domain=pe.Reals)
        var_name = 'C2'

        def rule(model, i, j):
            """
            C[i, j]==2 means that the pattern is matched, anything less is okay
            """
            C = operator.attrgetter(var_name)(model)[i, j]
            if self.task_chunk_min[j] <= chunk_len:
                return Constraint.Feasible
            return None, C, chunk_len - 1

        self.model.constrain_chunk20 = Constraint(self.model.c2timeslots,
                                                  self.model.tasks, rule=rule)

        def rule(model, i, j):
            C = operator.attrgetter(var_name)(model)[i, j]
            if self.task_chunk_min[j] <= chunk_len:
                return Constraint.Feasible
            total = sum(L[i, k] * model.A[k, j] for k in model.timeslots)
            return 0, C - total, None

        self.model.constrain_chunk21 = Constraint(self.model.c2timeslots,
                                                  self.model.tasks, rule=rule)

    def _get_chunk_parameters(self, chunk_len, offset, mode):
        """
        Helper method for chunking constraints

        :param chunk_len:
        :param offset:
        :param mode:
        :return:
        """
        filter = np.ones(chunk_len + offset * 2)
        if offset > 0:
            filter[0:offset] = -1
            filter[-offset:] = -1
        # print('XXX Chunk filter:', chunk_len, mode, filter)
        # filter = np.array([-1, 1, 1, 1, 1, 1, 1, -1])
        L, b = util.linop_from_1d_filter(filter, self.num_timeslots,
                                         offset=offset)
        c_len = self.num_timeslots - filter.size + 1 + offset * 2
        return filter, L, c_len

    def _get_rule_chunk_upper(self, mode, var_name, chunk_len, filter):
        """
        Helper method for chunking constraints

        :param mode:
        :param var_name:
        :param chunk_len:
        :param filter:
        :return:
        """

        def rule(model, i, j):
            """
            Upper bounds on filter match
            """
            C = operator.attrgetter(var_name)(model)[i, j]
            if mode == 'min':
                # For min mode, need to check that none of the smaller chunks
                # match (hence inequality)
                if self.task_chunk_min[j] <= chunk_len:
                    return Constraint.Feasible
                return None, C, chunk_len - 1
                # return None, model.C6m[i, j], chunk_len - 1
            elif mode == 'max':
                # For max mode, only need to check once (hence equality)
                if self.task_chunk_max[j] + 1 == chunk_len:
                    return None, C, filter.size - 1
                return Constraint.Feasible

        return rule

    def _get_rule_chunk_lower(self, mode, var_name, chunk_len, L):
        """
        Helper method for chunking constraints

        :param mode:
        :param var_name:
        :param chunk_len:
        :param L:
        :return:
        """

        def rule(model, i, j):
            """
            Lower bounds on filter match

            See CalendarSolver._get_rule_chunk_upper() for more details.
            """
            C = operator.attrgetter(var_name)(model)[i, j]
            if mode == 'min' and self.task_chunk_min[j] <= chunk_len:
                return Constraint.Feasible
            elif mode == 'max' and self.task_chunk_max[j] + 1 != chunk_len:
                return Constraint.Feasible
            total = sum(L[i, k] * model.A[k, j] for k in model.timeslots)
            return 0, C - total, None

        return rule

    def _constraints_chunking1m(self):
        """
        Ensures there are no unwanted 1-chunk tasks
        """
        chunk_len = 1
        mode = 'min'
        var_name = 'C1m'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c1mtimeslots = RangeSet(0, c_len - 1)
        self.model.C1m = Var(self.model.c1mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk1m0 = Constraint(self.model.c1mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk1m1 = Constraint(self.model.c1mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking2m(self):
        """
        Ensures there are no unwanted 2-chunk tasks
        """
        chunk_len = 2
        mode = 'min'
        var_name = 'C2m'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c2mtimeslots = RangeSet(0, c_len - 1)
        self.model.C2m = Var(self.model.c2mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk2m0 = Constraint(self.model.c2mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk2m1 = Constraint(self.model.c2mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking3m(self):
        """
        Ensures there are no unwanted 3-chunk tasks
        """
        chunk_len = 3
        mode = 'min'
        var_name = 'C3m'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c3mtimeslots = RangeSet(0, c_len - 1)
        self.model.C3m = Var(self.model.c3mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk3m0 = Constraint(self.model.c3mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk3m1 = Constraint(self.model.c3mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking4m(self):
        """
        Ensures there are no unwanted 4-chunk tasks
        """
        chunk_len = 4
        mode = 'min'
        var_name = 'C4m'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c4mtimeslots = RangeSet(0, c_len - 1)
        self.model.C4m = Var(self.model.c4mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk4m0 = Constraint(self.model.c4mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk4m1 = Constraint(self.model.c4mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking5m(self):
        """
        Ensures there are no unwanted 5-chunk tasks
        """
        chunk_len = 5
        mode = 'min'
        var_name = 'C5m'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c5mtimeslots = RangeSet(0, c_len - 1)
        self.model.C5m = Var(self.model.c5mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk5m0 = Constraint(self.model.c5mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk5m1 = Constraint(self.model.c5mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking6m(self):
        """
        Ensures there are no unwanted 6-chunk tasks
        """
        chunk_len = 6
        mode = 'min'
        var_name = 'C6m'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c6mtimeslots = RangeSet(0, c_len - 1)
        self.model.C6m = Var(self.model.c6mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk6m0 = Constraint(self.model.c6mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk6m1 = Constraint(self.model.c6mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking4M(self):
        """
        Ensures there are no unwanted 6+ chunk tasks

        FIXME(cathywu) this seems to slow down solving significantly.
        """
        chunk_len = 4
        mode = 'max'
        var_name = 'C4M'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c4Mtimeslots = RangeSet(0, c_len - 1)
        self.model.C4M = Var(self.model.c4Mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk4M0 = Constraint(self.model.c4Mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk4M1 = Constraint(self.model.c4Mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking5M(self):
        """
        Ensures there are no unwanted 6+ chunk tasks

        FIXME(cathywu) this seems to slow down solving significantly.
        """
        chunk_len = 5
        mode = 'max'
        var_name = 'C5M'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c5Mtimeslots = RangeSet(0, c_len - 1)
        self.model.C5M = Var(self.model.c5Mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk5M0 = Constraint(self.model.c5Mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk5M1 = Constraint(self.model.c5Mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking6M(self):
        """
        Ensures there are no unwanted 6+ chunk tasks

        FIXME(cathywu) this seems to slow down solving significantly.
        """
        chunk_len = 6
        mode = 'max'
        var_name = 'C6M'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c6Mtimeslots = RangeSet(0, c_len - 1)
        self.model.C6M = Var(self.model.c6Mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk6M0 = Constraint(self.model.c6Mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk6M1 = Constraint(self.model.c6Mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking7M(self):
        """
        Ensures there are no unwanted 6+ chunk tasks

        FIXME(cathywu) this seems to slow down solving significantly.
        """
        chunk_len = 7
        mode = 'max'
        var_name = 'C7M'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c7Mtimeslots = RangeSet(0, c_len - 1)
        self.model.C7M = Var(self.model.c7Mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk7M0 = Constraint(self.model.c7Mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk7M1 = Constraint(self.model.c7Mtimeslots,
                                                   self.model.tasks, rule=rule)

    def _constraints_chunking8M(self):
        """
        Ensures there are no unwanted 8+ chunk tasks

        FIXME(cathywu) this seems to slow down solving significantly.
        """
        chunk_len = 8
        mode = 'max'
        var_name = 'C8M'
        offset = 1 if mode == 'min' else 0

        filter, L, c_len = self._get_chunk_parameters(chunk_len, offset, mode)

        self.model.c8Mtimeslots = RangeSet(0, c_len - 1)
        self.model.C8M = Var(self.model.c8Mtimeslots * self.model.tasks,
                             domain=pe.Reals)

        rule = self._get_rule_chunk_upper(mode, var_name, chunk_len, filter)
        self.model.constrain_chunk8M0 = Constraint(self.model.c8Mtimeslots,
                                                   self.model.tasks, rule=rule)

        rule = self._get_rule_chunk_lower(mode, var_name, chunk_len, L)
        self.model.constrain_chunk8M1 = Constraint(self.model.c8Mtimeslots,
                                                   self.model.tasks, rule=rule)

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
            return self.task_duration[j] / self.task_chunk_max[j], switches, \
                   self.task_duration[j] / self.task_chunk_min[j]

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
        self._constraints_utility()
        self._constraints_task_valid()
        self._constraints_nonoverlapping_tasks()
        self._constraints_task_duration()
        # self._constraints_switching_bounds()
        self._constraints_task_contiguity()  # FIXME(cathywu) some slowdown

        self._constraints_chunking1m()
        self._constraints_chunking2m()
        self._constraints_chunking3m()
        self._constraints_chunking4m()
        self._constraints_chunking5m()
        self._constraints_chunking6m()

        # FIXME(cathywu) dramatic slowdown
        self._constraints_chunking4M()
        self._constraints_chunking5M()
        self._constraints_chunking6M()
        self._constraints_chunking7M()
        self._constraints_chunking8M()

        # objective
        self._objective_cost()
        # self._objective_switching()

    def optimize(self):
        # Create a model instance and optimize
        # self.instance = self.model.create_instance("data/calendar.dat")
        self.instance = self.model.create_instance()
        # See pyomo/opt/base/solvers.py:_presolve() for options
        self._results = self.opt.solve(self.instance, timelimit=TIMELIMIT,
                                       tee=True, keepfiles=True,
                                       report_timing=True)
        # self._results = self.opt.solve(self.instance, timelimit=2e2,
        # tee=True, keepfiles=True)
        self._optimized = True

    def display(self):
        self.instance.display()

    def visualize(self):
        """
        Visualization of calendar tasks, with hover for more details
        :return:
        """
        COLORS = d3['Category20'][20]

        array = np.reshape(
            [y for (x, y) in self.instance.A.get_values().items()],
            (self.num_timeslots, self.num_tasks))

        x, y = array.nonzero()
        top = (x % (24 * tutil.SLOTS_PER_HOUR)) / tutil.SLOTS_PER_HOUR
        bottom = top - (0.95 / tutil.SLOTS_PER_HOUR)
        left = np.floor(x / (24 * tutil.SLOTS_PER_HOUR))
        right = left + 0.95
        chunk_min = [self.task_chunk_min[k] for k in y]
        chunk_max = [self.task_chunk_max[k] for k in y]
        duration = [self.task_duration[k] for k in y]
        task = [self.task_names[k] for k in y]

        colors = [COLORS[i] for i in y]
        source = ColumnDataSource(
            data=dict(top=top, bottom=bottom, left=left, right=right,
                chunk_min=chunk_min, chunk_max=chunk_max, duration=duration,
                task_id=y, task=task, colors=colors, ))

        TOOLTIPS = [("task", "@task"), ("desc", "@task_id"),
            # ("(x,y)", "($x, $y)"),
            ("duration", "@duration"),
            ("chunk_range", "(@chunk_min, @chunk_max)"),
            ("(t,l)", "(@top, @left)"),
            # ("fill color", "$color[hex, swatch]:fill_color"),
            ("index", "$index"), ]

        # [Bokeh] inverted axis range example:
        # https://groups.google.com/a/continuum.io/forum/#!topic/bokeh/CJAvppgQmKo
        yr = Range1d(start=24.5, end=-0.5)
        xr = Range1d(start=-0.5, end=7.5)
        p = figure(plot_width=800, plot_height=800, y_range=yr, x_range=xr,
                   tooltips=TOOLTIPS, title="Calendar")
        self.p = p
        output_file("calendar.html")

        p.xaxis[0].axis_label = 'Weekday (Sun-Fri)'
        p.yaxis[0].axis_label = 'Hour (12AM-12AM)'

        # Replace default yaxis so that each hour is displayed
        p.yaxis[0].ticker.desired_num_ticks = 24
        p.yaxis[0].ticker.num_minor_ticks = 4
        p.xaxis[0].ticker.num_minor_ticks = 0

        # Display task allocation as colored rectangles
        p.quad(top='top', bottom='bottom', left='left', right='right',
               color='colors', source=source)

        # Pre-process task names for display (no repeats, abbreviated names)
        # FIXME(cathywu) currently assumes that y is in time order, which may
        #  not be the case when more task types are incorporated
        task_display = []
        curr_task = ""
        for name in task:
            if name == curr_task:
                task_display.append("")
            else:
                curr_task = name
                task_display.append(name)
        source2 = ColumnDataSource(
            data=dict(x=left, y=top, # abbreviated version of task
                task=[k[:17] for k in task_display], ))

        # Annotate rectangles with task name
        # [Bokeh] Text properties:
        # https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#text-properties
        labels = LabelSet(x='x', y='y', text='task', level='glyph', x_offset=3,
                          y_offset=-1, source=source2, text_font_size='7pt',
                          render_mode='canvas')
        p.add_layout(labels)

        show(p)
