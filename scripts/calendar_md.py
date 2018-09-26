"""
Script for CalendarSolver functionality with integrated markdown parser.
"""
import time

import numpy as np

from timealloc.calendar_solver import CalendarSolver
from timealloc.task_parser import TaskParser
import timealloc.util_time as tutil


DEFAULT_CHUNK_MIN = 2  # in IP slot units
DEFAULT_CHUNK_MAX = 20  # in IP slot units

time_allocation_fname = "scratch/time-allocation-2018-09-03.md"
tasks_fname = "scratch/tasks-2018-09-03.md"

tasks = TaskParser(time_allocation_fname, tasks_fname)

MODIFIERS = ['after', 'before', 'at', 'on']

# User defined parameters
num_tasks = len(tasks.tasks.keys())
num_timeslots = 24 * 7 * tutil.SLOTS_PER_HOUR

task_duration = np.ones(num_tasks)  # initialize task duration as 1 slot
task_chunk_min = DEFAULT_CHUNK_MIN * np.ones(num_tasks)
# FIXME(cathywu) 10 is currently not supported, so these constraints should be
#  off by default
task_chunk_max = DEFAULT_CHUNK_MAX * np.ones(num_tasks)

# FIXME(cathywu) have non-uniform utilities
utilities = np.ones((num_tasks, num_timeslots)).T

# Working hours
# TODO(cathywu) remove this for full scheduling version
stime = tutil.text_to_struct_time("8:30am")
work_mask = tutil.struct_time_to_slot_mask(stime, modifier="after")
stime = tutil.text_to_struct_time("9:30pm")
mask = tutil.struct_time_to_slot_mask(stime, modifier="before")
work_mask = np.array(np.logical_and(work_mask, mask), dtype=int)

print("Number of tasks", num_tasks)
overall_mask = np.ones((24*7*tutil.SLOTS_PER_HOUR, num_tasks))
for i, task in enumerate(tasks.tasks.keys()):
    total = tasks.tasks[task]["total"]
    task_duration[i] = tutil.hour_to_ip_slot(total)

    for key in tasks.tasks[task]:
        sub_mask = np.ones(24*7*tutil.SLOTS_PER_HOUR)
        if key in MODIFIERS:
            sub_mask = np.zeros(24*7*tutil.SLOTS_PER_HOUR)
            modifier = key
            attributes = tasks.tasks[task][key].split('; ')
            for attr in attributes:
                # print(task, key, attr)
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
            overall_mask[:, i] = np.array(
                np.logical_and(overall_mask[:,i], sub_mask), dtype=int)
        elif key == "chunks":
            chunks = tasks.tasks[task][key].split('-')
            task_chunk_min[i] = tutil.hour_to_ip_slot(float(chunks[0]))
            task_chunk_max[i] = tutil.hour_to_ip_slot(float(chunks[-1]))
        elif key == "total":
            pass
        else:
            print('Not yet handled key ({}) for {}'.format(key, task))

    # FIXME(cathywu) remove this later, this is for the "simplified IP"
    overall_mask[:, i] = np.array(np.logical_and(overall_mask[:,i], work_mask),
                                  dtype=int)
    # print(overall_mask.reshape((7,int(overall_mask.size/7))))

print('Chunks', task_chunk_min, task_chunk_max)

# Prepare the IP
params = {
    'num_tasks': num_tasks,
    'num_timeslots': num_timeslots,
    'task_duration': task_duration,
    'task_valid': overall_mask,
    'task_chunk_min': task_chunk_min,
    'task_chunk_max': task_chunk_max,
    'task_names': list(tasks.tasks.keys()),
}
cal = CalendarSolver(utilities, params)

# Solve
print("Optimizing...")
start_ts = time.time()
cal.optimize()
solve_time = time.time() - start_ts

# Display the results
cal.visualize()

cal.display()
array = np.reshape([y for (x,y) in cal.instance.A.get_values().items()],
                   (num_timeslots, num_tasks))
print("Schedule (timeslot x task):")
print(array)
print('Solve time', solve_time)
for i, task in enumerate(tasks.tasks.keys()):
    print(i, task)
