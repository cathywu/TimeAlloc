import numpy as np

from bokeh.palettes import d3
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LabelSet, Range1d
import pyomo.environ as pe
from pyomo.environ import AbstractModel, RangeSet, Var, Objective, Constraint, \
    summation, Expression
from pyomo.opt import SolverFactory

import timealloc.util as util
import timealloc.util_time as tutil
from timealloc.config_affinity import AFFINITY_COGNITIVE
from timealloc.util_time import NUMSLOTS

# For avoiding rounding issues
EPS = 1e-2  # epsilon

# Time limit for solver (wallclock)
TIMELIMIT = 500  # 3600, 1e3, 2e2, 50

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
        self.num_categories = params['num_categories']
        self.num_tasks = params['num_tasks']
        self.num_timeslots = params['num_timeslots']

        # num_tasks-by-num_categories matrix
        self.task_category = params['task_category']

        self.category_min = params['category_min']
        self.category_max = params['category_max']
        self.category_days = params['category_days']
        self.category_days_total = params['category_total']
        self.valid = params['task_valid']
        self.task_duration = params['task_duration']
        self.task_chunk_min = params['task_chunk_min']
        self.task_chunk_max = params['task_chunk_max']
        self.task_completion_bonus = params['task_completion_bonus']
        self.task_cognitive_load = params['task_cognitive_load']
        self.task_before = params['task_before']
        self.task_after = params['task_after']

        # Optional parameters
        if 'category_names' in params:
            self.cat_names = params['category_names']
        else:
            self.cat_names = ["C{}".format(k) for k in
                              range(self.num_categories)]

        if 'task_names' in params:
            self.task_names = params['task_names']
        else:
            self.task_names = ["T{}".format(i) for i in range(self.num_tasks)]

        if 'task_spread' in params:
            self.task_spread = params['task_spread']
        else:
            # Defaults to prefer contiguous scheduling
            self.task_spread = np.zeros(self.num_tasks)

        # Index sets for iteration
        self.model.tasks = RangeSet(0, self.num_tasks - 1)
        self.model.categories = RangeSet(0, self.num_categories - 1)
        self.model.timeslots = RangeSet(0, self.num_timeslots - 1)  # for A
        self.model.timeslots2 = RangeSet(0, self.num_timeslots - 2)  # for A2
        self.model.timeslots3 = RangeSet(0, self.num_timeslots - 3)  # for A3
        self.model.timeslots4 = RangeSet(0, self.num_timeslots - 4)  # for A4
        self.model.dtimeslots = RangeSet(0, self.num_timeslots - 2)

        # Fill pyomo Params from user params
        self.utilities = utilities

        # construct IP
        self._construct_ip()

        # Create a solver
        # self.opt = SolverFactory('glpk')
        # self.opt = SolverFactory('ipopt')
        # self.opt = SolverFactory('mosek')  # not available
        # self.opt = SolverFactory('cbc')
        self.opt = SolverFactory('cplex')
        # self.opt = SolverFactory('gurobi')
        # self.opt.options['tmlim'] = 1000  # glpk
        # self.opt.options['max_iter'] = 10000  # ipopt
        # self.opt.options['timelimit'] = 5

    def _variables(self):
        """
        Primary variables are defined here. Intermediate variables may be
        defined directly in constraints.

        Convention: variables are capitalized, i.e. model.A, and not model.a
        """
        # Allocation A
        self.model.A = Var(self.model.timeslots * self.model.tasks,
                           domain=pe.Boolean, initialize=0)
        # Total utility of allocation A
        self.model.A_total = Var(domain=pe.Reals)

        # Multi-resolution allocation (size 1-4 chunks)
        self.model.A2 = Var(self.model.timeslots * self.model.tasks,
                            domain=pe.Boolean, initialize=0)
        self.model.A2_total = Var(domain=pe.Reals)
        self.model.A3 = Var(self.model.timeslots * self.model.tasks,
                            domain=pe.Boolean, initialize=0)
        self.model.A3_total = Var(domain=pe.Reals)
        self.model.A4 = Var(self.model.timeslots * self.model.tasks,
                            domain=pe.Boolean, initialize=0)
        self.model.A4_total = Var(domain=pe.Reals)

        # Completion bonus
        self.model.T_total = Var(self.model.tasks, domain=pe.Integers,
                                 initialize=0)
        self.model.Completion_total = Var(domain=pe.Reals)

        self.model.Affinity_cognitive_total = Var(domain=pe.Reals)

        # Slots within a day
        self.model.intradayslots = RangeSet(0, self.num_timeslots/7-1)  # 7 days
        # Day slots
        self.model.dayslots = RangeSet(0, 6)  # 7 days
        # Tasks assigned on days
        self.model.S = Var(self.model.dayslots * self.model.tasks,
                           domain=pe.Integers, initialize=0)
        # Spread utility
        self.model.S_total = Var(domain=pe.Reals)

        # Task start/end slots (per day)
        self.model.T_end = Var(self.model.dayslots, self.model.tasks,
                               domain=pe.Integers,
                               bounds=(0, self.num_timeslots / 7 - 1))
        # self.model.T_start = Var(self.model.dayslots, self.model.tasks,
        #                          domain=pe.Integers,
        #                          bounds=(0, self.num_timeslots / 7 - 1))

        # Categories assigned on days
        self.model.S_cat = Var(self.model.dayslots * self.model.categories,
                               domain=pe.Boolean, initialize=0)
        # Total days on which categories are assigned
        self.model.S_cat_total = Var(self.model.categories, domain=pe.Integers)

        # Contiguity slots (half-days)
        self.cont_incr = CONT_STRIDE * tutil.SLOTS_PER_HOUR
        self.cont_slots = self.num_timeslots / self.cont_incr - 1
        self.model.contslots = RangeSet(0, self.cont_slots - 1)
        self.model.CTu = Var(self.model.contslots * self.model.tasks,
                             domain=pe.Integers, initialize=0)
        self.model.CTl = Var(self.model.contslots * self.model.tasks,
                             domain=pe.Integers, initialize=0)
        # Contiguity utility
        self.model.CTu_total = Var(domain=pe.Reals)
        self.model.CTl_total = Var(domain=pe.Reals)

        # Category durations
        self.model.C_total = Var(self.model.categories, domain=pe.Reals,
                                 initialize=0)

    def _objective_switching(self):
        """
        Reward task-specific amounts of task switching
        """
        # FIXME(cathywu) consider removing, not currently used

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
            total = model.A_total + model.A2_total + model.A3_total + \
                    model.A4_total
            total += model.Completion_total
            total += model.Affinity_cognitive_total
            total += model.CTu_total + model.CTl_total + model.S_total
            return -total

        # self.model.exp_cost = Expression(rule=obj_expression)
        # self.model.obj_cost = Objective(rule=self.model.exp_cost)
        self.model.obj_cost = Objective(rule=obj_expression)

    def _constraints_variables(self):
        """
        These are safe constraints because they are only used to define
        variables. So for debugging purposes, they can always be included.
        They should never be the cause of a constraint violation.
        """

        def rule(model, k):
            """
            Total slots allocated to category k
            """
            ind_i = model.timeslots
            ind_i2 = model.timeslots2
            ind_i3 = model.timeslots3
            ind_i4 = model.timeslots4
            ind_j = model.tasks
            cat_k_total = sum(
                model.A[i, j] * self.task_category[j, k] for i in ind_i for j in
                ind_j)
            cat_k_total += 2 * sum(
                model.A2[i, j] * self.task_category[j, k] for i in ind_i2 for j
                in ind_j)
            cat_k_total += 3 * sum(
                model.A3[i, j] * self.task_category[j, k] for i in ind_i3 for j
                in ind_j)
            cat_k_total += 4 * sum(
                model.A4[i, j] * self.task_category[j, k] for i in ind_i4 for j
                in ind_j)
            return model.C_total[k] == cat_k_total

        self.model.constrain_cat_duration0 = Constraint(self.model.categories,
                                                        rule=rule)

        def rule(model, s, k):
            """
            S_cat[s,k] = whether (any tasks of) category k is assigned on day s
            """
            den = sum(self.task_category[:, k])
            ind_j = model.tasks
            total = sum(self.task_category[j, k] * model.S[s, j] for j in
                        ind_j) / den
            # Desired: S[i,j] = ceil(total)
            # Desired: S[i,j] = 0 if total <= 0; otherwise, S[i,j] = 1
            return -EPS, model.S_cat[s, k] - total, 1 - EPS

        self.model.constrain_cat_days0 = Constraint(self.model.dayslots,
                                                    self.model.categories,
                                                    rule=rule)

        def rule(model, k):
            """
            S_cat_total[k] = number of unique days in which task from
            category k were assigned

            More precisely:
            sum_s S_cat[s,k] == S_cat_total[k]
            """
            ind_s = model.dayslots
            total = sum(model.S_cat[s, k] for s in ind_s)
            return model.S_cat_total[k] == total

        self.model.constrain_cat_days1 = Constraint(self.model.categories,
                                                    rule=rule)

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
        def rule(model):
            """
            Bind the tail entries to zero
            """
            num = self.num_timeslots
            ind_j = model.tasks
            total = sum(model.A2[num-1, j] for j in ind_j)
            total += sum(model.A3[num-1, j] for j in ind_j)
            total += sum(model.A4[num-1, j] for j in ind_j)
            total += sum(model.A3[num-2, j] for j in ind_j)
            total += sum(model.A4[num-2, j] for j in ind_j)
            total += sum(model.A4[num-3, j] for j in ind_j)
            return None, total, 0

        self.model.constrain_tail = Constraint(rule=rule)

        def rule(model):
            """
            Only permit "valid" allocation on A, A2, A3, etc.
            """
            ind_i = model.timeslots
            ind_j = model.tasks
            total = sum(model.A[i, j] * (1-self.valid[i, j]) for i in ind_i
                        for j in ind_j)
            total += sum(model.A2[i, j] * (1 - self.valid[i, j]) for i in
                         ind_i for j in ind_j)
            total += sum(model.A3[i, j] * (1 - self.valid[i, j]) for i in
                         ind_i for j in ind_j)

            return None, total, 0

        self.model.constrain_valid0 = Constraint(rule=rule)

        def rule(model):
            """
            Only permit "valid" allocation on A, A2, A3, etc.
            """
            ind_i = model.timeslots2
            ind_j = model.tasks
            inv = 1-self.valid
            total = sum(
                model.A2[i, j] * inv[i + 1, j] for i in ind_i for j in ind_j)
            total += sum(
                model.A3[i, j] * inv[i + 1, j] for i in ind_i for j in ind_j)
            total += sum(
                model.A4[i, j] * inv[i + 1, j] for i in ind_i for j in ind_j)

            ind_i = model.timeslots3
            ind_j = model.tasks
            total += sum(
                model.A3[i, j] * inv[i + 2, j] for i in ind_i for j in ind_j)
            total += sum(
                model.A4[i, j] * inv[i + 2, j] for i in ind_i for j in ind_j)

            ind_i = model.timeslots4
            ind_j = model.tasks
            total += sum(
                model.A4[i, j] * inv[i + 3, j] for i in ind_i for j in ind_j)

            return None, total, 0

        self.model.constrain_valid1 = Constraint(rule=rule)

    def _constraints_nonoverlapping_tasks(self):
        """
        No multi-tasking! No events should take place at the same time.
        Perhaps a future version can account for light-weight multi-tasking,
        like commuting + reading.
        """

        def rule(model, i):
            total = sum(model.A[i, j] for j in model.tasks)
            total += sum(model.A2[i, j] for j in model.tasks)
            total += sum(model.A3[i, j] for j in model.tasks)
            total += sum(model.A4[i, j] for j in model.tasks)
            if i > 0:
                total += sum(model.A2[i - 1, j] for j in model.tasks)
                total += sum(model.A3[i - 1, j] for j in model.tasks)
                total += sum(model.A4[i - 1, j] for j in model.tasks)
            if i > 1:
                total += sum(model.A3[i - 2, j] for j in model.tasks)
                total += sum(model.A4[i - 2, j] for j in model.tasks)
            if i > 2:
                total += sum(model.A4[i - 3, j] for j in model.tasks)
            return 0, total, 1

        self.model.constrain_nonoverlapping = Constraint(self.model.timeslots,
                                                         rule=rule)

    def _constraints_category_duration(self):
        """
        Each category duration should be within some user-specified range
        """

        def rule(model, k):
            return self.category_min[k], model.C_total[k], self.category_max[k]

        self.model.constrain_cat_duration1 = Constraint(self.model.categories,
                                                        rule=rule)

    def _constraints_task_duration(self):
        """
        Each task should stay within task-specific allocation bounds
        """

        def rule(model, j):
            task_j_total = sum(model.A[i, j] for i in model.timeslots)
            task_j_total += 2 * sum(model.A2[i, j] for i in model.timeslots2)
            task_j_total += 3 * sum(model.A3[i, j] for i in model.timeslots3)
            task_j_total += 4 * sum(model.A4[i, j] for i in model.timeslots4)
            return 0, task_j_total, self.task_duration[j]

        self.model.constrain_task_duration0 = Constraint(self.model.tasks,
                                                        rule=rule)

        def rule(model, j):
            """
            Task completion variables
            """
            task_j_total = sum(model.A[i, j] for i in model.timeslots)
            task_j_total += 2 * sum(model.A2[i, j] for i in model.timeslots2)
            task_j_total += 3 * sum(model.A3[i, j] for i in model.timeslots3)
            task_j_total += 4 * sum(model.A4[i, j] for i in model.timeslots4)
            task_j_completion = task_j_total / self.task_duration[j]
            return -1 + EPS, model.T_total[j] - task_j_completion, EPS

        self.model.constrain_task_duration1 = Constraint(self.model.tasks,
                                                        rule=rule)

    def _constraints_utility(self):
        """
        Utility sums for each allocation resolution
        """

        def rule(model):
            total = summation(self.utilities, model.A)
            return model.A_total == total

        self.model.constrain_A_total = Constraint(rule=rule)

        def rule(model):
            total = 2 * summation(self.utilities, model.A2)
            return model.A2_total == total

        self.model.constrain_A2_total = Constraint(rule=rule)

        def rule(model):
            total = 3 * summation(self.utilities, model.A3)
            return model.A3_total == total

        self.model.constrain_A3_total = Constraint(rule=rule)

        def rule(model):
            total = 4 * summation(self.utilities, model.A4)
            return model.A4_total == total

        self.model.constrain_A4_total = Constraint(rule=rule)

        def rule(model):
            completion_bonus = self.task_completion_bonus * self.task_duration
            total = summation(completion_bonus, model.T_total)
            return model.Completion_total == total

        self.model.constrain_completion_total = Constraint(rule=rule)

        def rule(model):
            scaling = 0.2
            affinity = np.outer(AFFINITY_COGNITIVE, self.task_cognitive_load)

            # TODO(cathywu) replace this code when "simple slicing" is clarified
            zeros1 = np.zeros((1, self.num_tasks))
            zeros2 = np.zeros((2, self.num_tasks))
            zeros3 = np.zeros((3, self.num_tasks))

            total = summation(affinity, model.A)
            total += summation(affinity, model.A2)
            total += summation(affinity, model.A3)
            total += summation(affinity, model.A4)

            total += summation(np.vstack((affinity[1:, :], zeros1)), model.A2)
            total += summation(np.vstack((affinity[1:, :], zeros1)), model.A3)
            total += summation(np.vstack((affinity[1:, :], zeros1)), model.A4)

            total += summation(np.vstack((affinity[2:, :], zeros2)), model.A3)
            total += summation(np.vstack((affinity[2:, :], zeros2)), model.A4)

            total += summation(np.vstack((affinity[3:, :], zeros3)), model.A4)
            total *= scaling

            return model.Affinity_cognitive_total == total

        self.model.constrain_affinity_cognitive_total = Constraint(rule=rule)

    def _constraints_category_days(self):
        """
        Ensure that enough (tasks of) categories are assigned on days as
        indicated by category_days.
        """

        def rule(model, s, k):
            """
            Ensure that a task of a category is assigned on each day as desired.

            More precisely:
            S_cat[s,k] >= cat_days[s, k]
            """
            return self.category_days[s, k], model.S_cat[s, k], None

        self.model.constrain_cat_days2 = Constraint(self.model.dayslots,
            self.model.categories, rule=rule)

        def rule(model, k):
            """
            Lower bound on number of distinct days in which a (task from a)
            category is assigned.

            More precisely:
            sum_s S_cat[s,k] = S_cat_total[k] >= cat_days[k]
            """
            return self.category_days_total[k], model.S_cat_total[k], None

        self.model.constrain_cat_days3 = Constraint(self.model.categories,
                                                    rule=rule)

    def _constraints_task_chunks(self):

        def rule(model, j):
            """
            Disable allocation at resolutions smaller than permitted chunk_min
            """
            ind_i = model.timeslots
            if self.task_chunk_min[j] == 2:
                total = sum(model.A[i, j] for i in ind_i)
                return None, total, 0
            elif self.task_chunk_min[j] == 3:
                total = sum(model.A[i, j] + model.A2[i, j] for i in ind_i)
                return None, total, 0
            elif self.task_chunk_min[j] >= 4:
                total = sum(
                    model.A[i, j] + model.A2[i, j] + model.A3[i, j] for i in
                    ind_i)
                return None, total, 0
            return Constraint.Feasible

        self.model.constrain_chunk_min = Constraint(self.model.tasks, rule=rule)

        def rule(model, j):
            """
            Disable allocation at resolutions larger than permitted chunk_max
            """
            ind_i = model.timeslots
            if self.task_chunk_max[j] <= 1:
                total = sum(
                    model.A2[i, j] + model.A3[i, j] + model.A4[i, j] for i in
                    ind_i)
                return None, total, 0
            elif self.task_chunk_max[j] <= 2:
                total = sum(model.A3[i, j] + model.A4[i, j] for i in ind_i)
                return None, total, 0
            elif self.task_chunk_max[j] <= 3:
                total = sum(model.A4[i, j] for i in ind_i)
                return None, total, 0
            return Constraint.Feasible

        self.model.constrain_chunk_max = Constraint(self.model.tasks, rule=rule)

    def _constraints_dependencies(self):
        """
        Before/after task dependencies
        """
        # FIXME(cathywu) returning infeasible solutions, de-prioritized for now

        def rule(model, d, i, j):
            """
            A[i + off] * num[i] <= T_end[d, j]
            """
            off = d * 48
            num = np.arange(self.num_timeslots / 7)
            total = model.A[i + off, j] * num[i]
            return 0, model.T_end[d, j] - total, None

        self.model.constrain_task_end = Constraint(self.model.dayslots,
                                                   self.model.intradayslots,
                                                   self.model.tasks, rule=rule)

        self.model.T0_end = Var(self.model.timeslots, self.model.tasks,
                                domain=pe.Boolean)

        def rule(model, d, i, j):
            """
            T0_end[i + off,j] = 1 iff i <= T_end[d, j]
            """
            off = d * 48
            num = np.arange(self.num_timeslots / 7)
            total = (model.T_end[d, j] - num[i] + 1) / 48
            return -EPS, model.T0_end[i + off, j] - total, 1 - EPS

        self.model.constrain_task_end1 = Constraint(self.model.dayslots,
                                                    self.model.intradayslots,
                                                    self.model.tasks, rule=rule)

        def rule(model, i, j):
            """
            Entries A[i, j] after T_end[d, j] are zero.
            That is, either the entry is before T_end[d, j] or it is zero.

            T0_end[i,j] == 1 or A[i,j] == 0
            """
            return 1, model.T0_end[i, j] + (1-model.A[i, j]), None

        # self.model.constrain_task_end2 = Constraint(self.model.timeslots,
        #                                            self.model.tasks, rule=rule)

        def rule(model, d, j0, j1):
            """
            If j0 should be after j1, then numbers[i0] >= numbers[i1]
            Of course, this only applies to active slots in A.

            model.T_end[d, j0] >= model.T_end[d, j1]
            """
            if self.task_after[j0, j1] == 0:
                return Constraint.Feasible
            total = model.T_end[d, j0] - model.T_end[d, j1]
            return 0, total, None

        self.model.constrain_after = Constraint(self.model.dayslots,
                                                self.model.tasks,
                                                self.model.tasks, rule=rule)

    def _constraints_task_spread(self):
        """
        Encourage the chunks of a task to be spread out. In particular,
        reward the number of days that a task is scheduled.
        """
        # encourage scheduling a chunk for every 24 hours
        incr = 24 * tutil.SLOTS_PER_HOUR
        diag = util.blockdiag(self.num_timeslots, incr=incr)
        slots = diag.shape[0]

        def rule(model, p, j):
            """
            For spread-activated tasks, this rule is used to encourage
            spreading the chunks out on multiple days.

            More precisely:
            S[i,j] = whether task j is assigned on day i

            Maximizing sum_i S[i,j] encourages spreading out the task chunks
            """
            den = sum(diag[p, :])
            ind_i = model.timeslots
            total = sum(diag[p, i] * (
                model.A[i, j] + 2 * model.A2[i, j] + 3 * model.A3[i, j] + 4 *
                model.A4[i, j]) for i in ind_i)
            total /= den
            # Desired: S[i,j] = ceil(total)
            # Desired: S[i,j] = 0 if total <= 0; otherwise, S[i,j] = 1
            return -EPS, model.S[p, j] - total, 1 - EPS

        self.model.constrain_spread0 = Constraint(self.model.dayslots,
                                                  self.model.tasks, rule=rule)

        def rule(model):
            den = self.num_tasks * slots
            num = 20
            weights = np.ones((7, self.num_tasks))
            for j in range(self.num_tasks):
                weights[:, j] = self.task_spread[j]
            total = summation(weights, model.S) / den * num
            return model.S_total == total

        self.model.constrain_spread1 = Constraint(rule=rule)

    def _constraints_task_contiguity(self):
        """
        Encourage the chunks of a tasks to be scheduled close to one another,
        i.e. reward shorter "elapsed" times
        """
        # CONT_STRIDE=1 would give original implementation
        triu = util.triu(self.num_timeslots, incr=self.cont_incr)
        tril = util.tril(self.num_timeslots, incr=self.cont_incr)

        def rule(model, i, j):
            """
            This rule is used to encourage early completion (in terms of
            allocation) of a task.

            More precisely:
            CTu[i,j] = whether task j is UNASSIGNED between slot i and the end

            Maximizing sum_i CTu[i,j] encourages early task completion.
            Maximizing sum_i CTu[i,j]+CTl[i,j] encourages contiguous scheduling.
            """
            active = 1-self.task_spread[j]
            den = sum(triu[i, :])
            ind = model.timeslots
            # FIXME(cathywu) can/should be more precise with A,A2,A3 offsets
            total = sum(triu[i, k] * (
                1 - model.A[k, j] - model.A2[k, j] - model.A3[k, j] - model.A4[
                    k, j]) for k in ind)
            total /= den
            total *= active
            # CTu[i,j] = floor(total)
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
            active = 1-self.task_spread[j]
            den = sum(tril[i, :])
            ind = model.timeslots
            total = sum(tril[i, k] * (
                1 - model.A[k, j] - model.A2[k, j] - model.A3[k, j] - model.A4[
                    k, j]) for k in ind)
            total /= den
            total *= active
            return -1 + EPS, model.CTl[i, j] - total, EPS + self.slack_cont

        self.model.constrain_contiguity_l = Constraint(self.model.contslots,
                                                       self.model.tasks,
                                                       rule=rule)

        def rule(model):
            den = self.num_tasks * self.cont_slots * (self.slack_cont + 1)
            num = 0.25
            total = summation(model.CTu) / den * num
            return model.CTu_total == total

        self.model.constrain_contiguity_ut = Constraint(rule=rule)

        def rule(model):
            den = self.num_tasks * self.cont_slots * (self.slack_cont + 1)
            num = 0.25
            total = summation(model.CTl) / den * num
            return model.CTl_total == total

        self.model.constrain_contiguity_lt = Constraint(rule=rule)

    def _construct_ip(self):
        """
        Aggregates MIP construction
        """
        # name the problem
        self.integer_program = "CalenderSolver"
        # variables
        self._variables()
        # constraints
        self._constraints_variables()
        self._constraints_external()
        self._constraints_other()
        self._constraints_utility()
        self._constraints_category_duration()
        self._constraints_task_valid()
        self._constraints_nonoverlapping_tasks()
        self._constraints_task_duration()
        self._constraints_task_contiguity()  # FIXME(cathywu) some slowdown
        self._constraints_task_spread()
        self._constraints_category_days()
        self._constraints_task_chunks()  # imposes chunk bounds

        # FIXME this might be horrendously slow
        # self._constraints_dependencies()  # de-prioritized

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
        self._collect_results()
        # self._results = self.opt.solve(self.instance, timelimit=2e2,
        # tee=True, keepfiles=True)
        self._optimized = True

    def _collect_results(self):
        array1 = np.reshape(
            [y for (x, y) in self.instance.A.get_values().items()],
            (self.num_timeslots, self.num_tasks))
        self.array1 = np.round(array1)

        array2 = np.reshape(
            [y for (x, y) in self.instance.A2.get_values().items()],
            (self.num_timeslots, self.num_tasks))
        self.array2 = np.round(array2)

        array3 = np.reshape(
            [y for (x, y) in self.instance.A3.get_values().items()],
            (self.num_timeslots, self.num_tasks))
        self.array3 = np.round(array3)

        array4 = np.reshape(
            [y for (x, y) in self.instance.A4.get_values().items()],
            (self.num_timeslots, self.num_tasks))
        self.array4 = np.round(array4)

        array = self.array1.copy()
        array += self.array2
        array[1:, :] += self.array2[:-1, :]
        array += self.array3
        array[1:, :] += self.array3[:-1, :]
        array[2:, :] += self.array3[:-2, :]
        array += self.array4
        array[1:, :] += self.array4[:-1, :]
        array[2:, :] += self.array4[:-2, :]
        array[3:, :] += self.array4[:-3, :]
        self.array = array
        # Task realizations
        self.task_duration_realized = self.array.sum(axis=0)
        # Category realizations
        self.category_duration_realized = np.array(
            [y for (x, y) in self.instance.C_total.get_values().items()])

        self.affinity = np.outer(AFFINITY_COGNITIVE, self.task_cognitive_load)

    def display(self):
        # self.instance.display()  # Displays everything
        self.instance.A_total.display()
        self.instance.A2_total.display()
        self.instance.A3_total.display()
        self.instance.A4_total.display()
        self.instance.Completion_total.display()
        self.instance.Affinity_cognitive_total.display()
        self.instance.S_total.display()
        self.instance.CTu_total.display()
        self.instance.CTl_total.display()
        self.instance.S_cat_total.display()
        self.instance.C_total.display()

    def get_diagnostics(self):
        # Display task realizations (ordered by decreasing task_duration)
        print("Task realizations:")
        task_sort_ind = np.argsort(self.task_duration)[::-1]
        for i in task_sort_ind:
            if self.task_duration[i] != self.task_duration_realized[i] and \
                            self.task_duration[i] != NUMSLOTS:
                print('{:3.0f} [{:3.0f}] {} ({}) INCOMPLETE'.format(
                    self.task_duration_realized[i], self.task_duration[i],
                    self.task_names[i], i))
            else:
                print('{:3.0f} [{:3.0f}] {} ({})'.format(
                    self.task_duration_realized[i], self.task_duration[i],
                    self.task_names[i], i))
        # Display category realizations (ordered by decreasing category_min)
        print("Category realizations:")
        cat_sort_ind = np.argsort(self.category_min)[::-1]
        for i in cat_sort_ind:
            if self.category_min[i] != self.category_duration_realized[i]:
                print('{:3.0f} [{:3.0f}, {:3.0f}] {} ({}) EXTRA'.format(
                    self.category_duration_realized[i], self.category_min[i],
                    self.category_max[i], self.cat_names[i], i))
            else:
                print('{:3.0f} [{:3.0f}, {:3.0f}] {} ({})'.format(
                    self.category_duration_realized[i], self.category_min[i],
                    self.category_max[i], self.cat_names[i], i))

    def visualize(self):
        """
        Visualization of calendar tasks, with hover for more details
        """
        COLORS = d3['Category20c'][20] + d3['Category20b'][20]
        COLORS_CAT = d3['Category20'][20]

        times, tasks = self.array.nonzero()
        bottom = (times % (24 * tutil.SLOTS_PER_HOUR)) / tutil.SLOTS_PER_HOUR
        top = bottom + (0.95 / tutil.SLOTS_PER_HOUR)
        left = np.floor(times / (24 * tutil.SLOTS_PER_HOUR))
        right = left + 0.95
        chunk_min = [self.task_chunk_min[k] for k in tasks]
        chunk_max = [self.task_chunk_max[k] for k in tasks]
        affinity_cog_task = [self.task_cognitive_load[j] for j in tasks]
        affinity_cog_slot = [AFFINITY_COGNITIVE[i] for i in times]
        affinity_cognitive = (np.array(affinity_cog_task) * np.array(
            affinity_cog_slot)).tolist()
        duration = [self.task_duration[k] for k in tasks]
        task_names = [self.task_names[k] for k in tasks]
        category_ids = [[l for l, j in enumerate(array) if j != 0] for array in
                        [self.task_category[j, :] for j in tasks]]
        category = [", ".join(
            [self.cat_names[l] for l, j in enumerate(array) if j != 0]) for
                    array in [self.task_category[j, :] for j in tasks]]

        offset = self.num_tasks - self.num_categories
        # Use #deebf7 as placeholder/default event color
        colors = [COLORS[i % len(COLORS)] if i < offset else '#ffffcc' for i in
                  tasks]
        source = ColumnDataSource(data=dict(
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            chunk_min=chunk_min,
            chunk_max=chunk_max,
            affinity_cognitive=affinity_cognitive,
            affinity_cog_slot=affinity_cog_slot,
            affinity_cog_task=affinity_cog_task,
            duration=duration,
            task_id=tasks,
            task=task_names,
            category=category,
            colors=colors,
        ))

        TOOLTIPS = [("task", "@task"),
                    ("desc", "@task_id"),
                    ("category", "@category"),
                    ("duration", "@duration"),
                    ("chunk_range", "(@chunk_min, @chunk_max)"),
                    ("affinity [slot x task]", "@affinity_cognitive = "
                                               "@affinity_cog_slot x "
                                               "@affinity_cog_task"),
                    ("(t,l)", "(@top, @left)"),
                    ("index", "$index"),
                    ]

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
        for name in task_names:
            if name == curr_task:
                task_display.append("")
            else:
                curr_task = name
                task_display.append(name)
        source2 = ColumnDataSource(data=dict(
            x=left,
            y=top,  # abbreviated version of task
            task=[k[:17] for k in task_display],
        ))

        # Annotate rectangles with task name
        # [Bokeh] Text properties:
        # https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#text-properties
        labels = LabelSet(x='x', y='y', text='task', level='glyph', x_offset=3,
                          y_offset=-1, source=source2, text_font_size='7pt',
                          render_mode='canvas')
        p.add_layout(labels)

        # Display categories as a colored line on the left
        # TODO(cathywu) currently displays only the "first" category,
        # add support for more categories
        xs = []
        ys = []
        for y0, y1, x in zip(top, bottom, left):
            xs.append([x, x])
            ys.append([y0, y1])

        colors_cat = [COLORS_CAT[cat_ids[0] % len(COLORS_CAT)] for cat_ids in
                      category_ids]
        source3 = ColumnDataSource(data=dict(
            xs=xs,
            ys=ys,
            colors=colors_cat,
        ))
        p.multi_line(xs='xs', ys='ys', color='colors', line_width=4,
                     source=source3)

        show(p)
