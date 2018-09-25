"""
Script for CalendarSolver functionality with integrated markdown parser.
"""
import time

import numpy as np

from timealloc.calendar_solver import CalendarSolver
from timealloc.task_parser import TaskParser
import timealloc.util_time as tutil


time_allocation_fname = "scratch/time-allocation-2018-09-03.md"
tasks_fname = "scratch/tasks-2018-09-03.md"

tasks = TaskParser(time_allocation_fname, tasks_fname)

MODIFIERS = ['after', 'before', 'at', 'on']

num_tasks = len(tasks.tasks.keys())
print("Number of tasks", num_tasks)
for task in tasks.tasks.keys():
    overall_mask = np.ones(24*7*tutil.SLOTS_PER_HOUR)
    total = tasks.tasks[task]["total"]
    for key in tasks.tasks[task]:
        print(task, key)
        sub_mask = np.ones(24*7*tutil.SLOTS_PER_HOUR)
        if key in MODIFIERS:
            sub_mask = np.zeros(24*7*tutil.SLOTS_PER_HOUR)
            modifier = key
            attributes = tasks.tasks[task][key].split('; ')
            for attr in attributes:
                print(task, key, attr)
                try:
                    stime = tutil.text_to_struct_time(attr)
                    mask = tutil.struct_time_to_slot_mask(stime,
                                                          modifier=modifier,
                                                          duration=tutil.hour_to_ip_slot(
                                                              total))
                except UnboundLocalError:
                    try:
                        dtime = tutil.text_to_datetime(attr, weekno=39,
                                                       year=2018)
                        mask = tutil.datetime_to_slot_mask(dtime,
                                                           modifier=modifier,
                                                           duration=tutil.hour_to_ip_slot(
                                                               total))
                    except UnboundLocalError:
                        raise (NotImplementedError,
                               "{} {} not supported".format(modifier, attr))
                sub_mask = np.logical_or(sub_mask, mask)
        overall_mask = np.array(np.logical_and(overall_mask, sub_mask),
                                dtype=int)
    print(overall_mask.reshape((7, 96)))


# User defined parameters
num_tasks = 5
num_timeslots = 30
task_duration = 6*np.ones(num_tasks)

utilities = np.ones((num_tasks, num_timeslots)).T
utilities[0, 0] = 0  # forces entry (0,0) to be unassigned
task_chunk_min = 2*np.ones(num_tasks)
task_chunk_min[0] = 1
task_chunk_max = 5*np.ones(num_tasks)
task_chunk_max[3] = 6

# Prepare the IP
params = {
    'num_tasks': num_tasks,
    'num_timeslots': num_timeslots,
    'task_duration': task_duration,
    'task_chunk_min': task_chunk_min,
    'task_chunk_max': task_chunk_max,
}
cal = CalendarSolver(utilities, params)

# Solve
start_ts = time.time()
# cal.optimize()
solve_time = time.time() - start_ts

# Display the results
# cal.display()
# array = np.reshape([y for (x,y) in cal.instance.A.get_values().items()],
#                    (num_timeslots, num_tasks))
# print("Schedule (timeslot x task):")
# print(array)
# print('Solve time', solve_time)
