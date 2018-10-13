import numpy as np

from timealloc.task_parser import TaskParser
from timealloc.util_time import NUMSLOTS
import timealloc.util_time as tutil
import timealloc.config_affinity as c

DEFAULT_CHUNK_MIN = 2  # in IP slot units
DEFAULT_CHUNK_MAX = 20  # in IP slot units


class Tasks:

    def __init__(self, time_allocation_fname, tasks_fname):
        self.tasks = TaskParser(time_allocation_fname, tasks_fname)

        # There are 3 types of tasks: work, other, and category defaults
        self.work_task_names = list(self.tasks.work_tasks.keys())
        self.num_work_tasks = len(self.work_task_names)
        self.other_task_names = list(self.tasks.other_tasks.keys())
        self.num_other_tasks = len(self.other_task_names)
        self.category_names = list(self.tasks.time_alloc.keys())
        self.num_categories = len(self.category_names)

        # Collect all the task names
        self.task_names = self.work_task_names
        self.task_names += self.other_task_names
        # use category name as default task name
        self.task_names += self.category_names
        self.num_tasks = self.num_work_tasks + self.num_other_tasks + \
                         self.num_categories

        # Plan from now
        # TODO(cathywu) specify a plan from time
        # self.start = datetime.today()
        # FIXME(cathywu) partial implementation
        # weekday = (self.start.weekday() + 2) % 7
        # offset = weekday * tutil.SLOTS_PER_DAY + self.start.hour * \
        #                                          tutil.SLOTS_PER_HOUR

        self._default_attributes()
        self._read_category_attributes()
        self._read_other_task_attributes()
        self._read_work_task_attributes()
        self._permit_short_tasks()
        self._shift_category_days()
        self._checks()

    def _default_attributes(self):
        # TODO clean up
        # initialize task duration as 1 slot
        self.task_duration = NUMSLOTS * np.ones(self.num_tasks)
        self.task_chunk_min = DEFAULT_CHUNK_MIN * np.ones(self.num_tasks)
        # FIXME(cathywu) 10 is currently not supported, so these constraints should be
        #  off by default
        self.task_chunk_max = DEFAULT_CHUNK_MAX * np.ones(self.num_tasks)
        
        # Special setup for default tasks (default task for each category)
        # these tasks can be slotted in however
        self.task_chunk_min[self.num_work_tasks:] = 0
        self.task_chunk_max[self.num_work_tasks:] = NUMSLOTS
        
        # num_tasks-by-num_tasks matrices
        # FIXME(cathywu) currently not used
        self.before = np.zeros((self.num_tasks, self.num_tasks))
        self.after = np.zeros((self.num_tasks, self.num_tasks))
        
        # num_tasks-by-num_categories matrix
        self.task_category = np.zeros((self.num_tasks, self.num_categories))
        self.category_min = np.zeros(self.num_categories)
        self.category_max = NUMSLOTS * np.ones(self.num_categories)
        
        # FIXME(cathywu) have non-uniform utilities
        self.utilities = 0.5 * np.ones((NUMSLOTS, self.num_tasks))
        # Fewer points for scheduling 'other' tasks
        # TODO parameterize this
        self.utilities[:, self.num_work_tasks:] = 0.333
        # Fewer points for scheduling default tasks
        # TODO parameterize this
        self.utilities[:, self.num_work_tasks + self.num_other_tasks:] = 0

        # Completion bonus for fully scheduling tasks
        self.completion_bonus = 0.5 * np.ones(self.num_tasks)
        self.completion_bonus[self.num_work_tasks:] = 0.333
        self.completion_bonus[self.num_work_tasks + self.num_other_tasks:] = 0
        
        # Cognitive load for each task [-1, 1]
        self.cognitive_load = np.zeros(self.num_tasks)
        
        # contiguous (0) or spread (1) scheduling; default is contiguous (0)
        self.task_spread = np.zeros(self.num_tasks)
        # by default, categories are allowed to be assigned on any timeslots
        self.category_masks = np.ones((NUMSLOTS, self.num_categories))
        # by default, no categories need to be assigned on any particular days
        self.category_days = np.zeros((7, self.num_categories))
        # by default, each category is required on 0 days
        self.category_days_total = np.zeros(self.num_categories)
        
        # Task specific time constraints mask
        # Assume first num_work_tasks entries are for work entries
        self.overall_mask = np.ones((NUMSLOTS, self.num_tasks))

    def _read_category_attributes(self):
        """
        CATEGORIES
        Read out per-category attributes
        """
        offset = self.num_work_tasks + self.num_other_tasks
        for k, cat in enumerate(self.category_names):
            # categories for default tasks
            self.task_category[offset + k, k] = 1

            for key in self.tasks.time_alloc[cat]:
                if key == "when":
                    for clause in self.tasks.time_alloc[cat][key]:
                        sub_mask = tutil.modifier_mask(clause, start=c.START,
                                                       total=self.category_min[
                                                           k], weekno=c.WEEKNO,
                                                       year=c.YEAR)
                        self.category_masks[:, k] = np.array(
                            np.logical_and(self.category_masks[:, k], sub_mask),
                            dtype=int)
                elif key == "chunks":
                    chunks = self.tasks.time_alloc[cat][key].split('-')
                    self.task_chunk_min[offset + k] = tutil.hour_to_ip_slot(
                        float(chunks[0]))
                    self.task_chunk_max[offset + k] = tutil.hour_to_ip_slot(
                        float(chunks[-1]))
                elif key == "total":
                    pass
                elif key == "min":
                    self.category_min[k] = self.tasks.time_alloc[cat][
                                               key] * tutil.SLOTS_PER_HOUR
                elif key == "max":
                    self.category_max[k] = self.tasks.time_alloc[cat][
                                               key] * tutil.SLOTS_PER_HOUR
                elif key == "days":
                    self.category_days[:, k], self.category_days_total[
                        k] = tutil.parse_days(self.tasks.time_alloc[cat][key])
                elif key == "cognitive load":
                    self.cognitive_load[offset + k] = float(
                        self.tasks.time_alloc[cat][key])
                elif key == "before":
                    other_task = self.tasks.time_alloc[cat][key]
                    other_task_ind = self.category_names.index(other_task)
                    self.before[offset + k, offset + other_task_ind] = 1
                elif key == "after":
                    other_task = self.tasks.time_alloc[cat][key]
                    other_task_ind = self.category_names.index(other_task)
                    self.after[offset + k, offset + other_task_ind] = 1
                else:
                    print('Not yet handled key ({}) for {}'.format(key, cat))
        self.overall_mask[:, -self.num_categories:] = self.category_masks

    def _read_other_task_attributes(self):
        """
        OTHER TASKS
        """
        offset = self.num_work_tasks
        for i, task in enumerate(self.other_task_names):
            total = self.tasks.other_tasks[task]["total"]
            self.task_duration[offset + i] = tutil.hour_to_ip_slot(total)

            for key in self.tasks.other_tasks[task]:
                if key == "when":
                    for clause in self.tasks.other_tasks[task][key]:
                        sub_mask = tutil.modifier_mask(clause, start=c.START,
                                                       total=total,
                                                       weekno=c.WEEKNO,
                                                       year=c.YEAR)
                        self.overall_mask[:, offset + i] = np.array(
                            np.logical_and(self.overall_mask[:, offset + i],
                                           sub_mask), dtype=int)
                elif key == "categories":
                    categories = self.tasks.other_tasks[task][key]
                    for cat in categories:
                        cat_id = self.category_names.index(cat)
                        self.task_category[offset + i, cat_id] = 1
                        category_mask = self.category_masks[:, cat_id]
                        self.overall_mask[:, offset + i] = np.array(
                            np.logical_and(self.overall_mask[:, offset + i],
                                           category_mask), dtype=int)
                elif key == "chunks":
                    chunks = self.tasks.other_tasks[task][key].split('-')
                    self.task_chunk_min[offset + i] = tutil.hour_to_ip_slot(
                        float(chunks[0]))
                    self.task_chunk_max[offset + i] = tutil.hour_to_ip_slot(
                        float(chunks[-1]))
                elif key == "total":
                    pass
                elif key == 'important':
                    self.utilities[:, offset + i] += 3
                elif key == 'urgent':
                    self.utilities[:, offset + i] += 10
                elif key == 'spread':
                    self.task_spread[offset + i] = True
                elif key == "cognitive load":
                    self.cognitive_load[offset + i] = float(
                        self.tasks.other_tasks[task][key])
                elif key == 'completion':
                    if self.tasks.other_tasks[task][key] == 'off':
                        self.completion_bonus[offset + i] = 0
                        self.utilities[:, offset + i] = 0.667
                elif key == 'display name':
                    # Use tasks display names if provided
                    # TODO(cathywu) Use full task names for eventual gcal events?
                    self.task_names[offset + i] = self.tasks.other_tasks[task][
                        'display name']
                else:
                    print('Not yet handled key ({}) for {}'.format(key, task))

    def _read_work_task_attributes(self):
        """
        WORK TASKS
        Working hours
        """
        work_category = self.category_names.index("Work")
        work_mask = self.category_masks[:, work_category]
        work_tasks = range(self.num_work_tasks)
        
        print("Number of tasks", self.num_tasks)
        # Task specific time constraints mask
        offset = 0
        for i, task in enumerate(self.tasks.work_tasks.keys()):
            total = self.tasks.work_tasks[task]["total"]
            self.task_duration[i] = tutil.hour_to_ip_slot(total)
            # toggle work category
            self.task_category[offset + i, work_category] = 1

            for key in self.tasks.work_tasks[task]:
                if key == "when":
                    for clause in self.tasks.work_tasks[task][key]:
                        sub_mask = tutil.modifier_mask(clause, start=c.START,
                                                       total=total,
                                                       weekno=c.WEEKNO,
                                                       year=c.YEAR)
                        self.overall_mask[:, i] = np.array(
                            np.logical_and(self.overall_mask[:, i], sub_mask),
                            dtype=int)
                elif key == "categories":
                    # other categories
                    categories = self.tasks.work_tasks[task][key]
                    for cat in categories:
                        cat_id = self.category_names.index(cat)
                        self.task_category[offset + i, cat_id] = 1
                        category_mask = self.category_masks[:, cat_id]
                        self.overall_mask[:, offset + i] = np.array(
                            np.logical_and(self.overall_mask[:, offset + i],
                                           category_mask), dtype=int)
                elif key == "chunks":
                    chunks = self.tasks.work_tasks[task][key].split('-')
                    self.task_chunk_min[i] = tutil.hour_to_ip_slot(
                        float(chunks[0]))
                    self.task_chunk_max[i] = tutil.hour_to_ip_slot(
                        float(chunks[-1]))
                elif key == "total":
                    pass
                elif key == 'soon':
                    for k in range(tutil.LOOKAHEAD):
                        weight = 0.5 / tutil.LOOKAHEAD
                        start = k*tutil.SLOTS_PER_DAY
                        end = (k+1)*tutil.SLOTS_PER_DAY
                        self.utilities[start:end, offset + i] += weight * (
                            tutil.LOOKAHEAD - k)
                elif key == 'important':
                    self.utilities[:, offset + i] += 3
                elif key == 'urgent':
                    self.utilities[:, offset + i] += 10
                elif key == 'spread':
                    self.task_spread[i] = True
                elif key == "cognitive load":
                    self.cognitive_load[offset + i] = float(
                        self.tasks.work_tasks[task][key])
                elif key == 'completion':
                    if self.tasks.work_tasks[task][key] == 'off':
                        self.completion_bonus[offset + i] = 0
                        self.utilities[:, offset + i] = 1
                elif key == 'display name':
                    # Use tasks display names if provided
                    # TODO(cathywu) Use full task names for eventual gcal events?
                    self.task_names[i] = self.tasks.work_tasks[task][
                        'display name']
                else:
                    print('Not yet handled key ({}) for {}'.format(key, task))

            self.overall_mask[:, i] = np.array(
                np.logical_and(self.overall_mask[:, i], work_mask), dtype=int)
            # print(overall_mask.reshape((7,int(overall_mask.size/7))))

    def _permit_short_tasks(self):
        """
        Permit the scheduling of short tasks
        """
        # TODO(cathywu) Permit the grouping of small tasks into larger ones? Like an
        # errands block
        for i in range(self.num_tasks):
            if self.task_chunk_min[i] > self.task_duration[i]:
                self.task_chunk_min[i] = self.task_duration[i]

    def _shift_category_days(self):
        if c.START is None:
            self.category_days_lookahead = self.category_days
        else:
            day = (c.START.weekday() + 2) % 7
            self.category_days_lookahead = np.vstack(
                (self.category_days[day:, :], self.category_days[:day, :]))

    def get_ip_params(self):
        # Prepare the IP
        params = {
            'num_timeslots': NUMSLOTS,
            'num_categories': self.num_categories,
            'category_names': self.category_names,
            'category_min': self.category_min,
            'category_max': self.category_max,
            'category_days': self.category_days_lookahead,  # e.g. M T W Sa Su
            'category_total': self.category_days_total,  # e.g. 3 of 5 days
            'task_category': self.task_category,
            'num_tasks': self.num_tasks,
            'task_duration': self.task_duration,
            'task_valid': self.overall_mask,
            'task_chunk_min': self.task_chunk_min,
            'task_chunk_max': self.task_chunk_max,
            'task_names': self.task_names,
            'task_spread': self.task_spread,
            'task_completion_bonus': self.completion_bonus,
            'task_cognitive_load': self.cognitive_load,
            'task_before': self.before,
            'task_after': self.after,
        }
        return params

    def display(self):
        print("All task names:")
        for i, task in enumerate(self.task_names):
            print(i, task)
        
        print("Category min/max:")
        print(self.category_min)
        print(self.category_max)
        
        print('Chunks min/max:')
        print(self.task_chunk_min)
        print(self.task_chunk_max)

    def _checks(self):
        # Assert that all tasks have at least 1 category
        error = "There are tasks without categories"
        assert np.prod(self.task_category.sum(axis=1)) > 0, error
