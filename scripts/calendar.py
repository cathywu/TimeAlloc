"""
Script for basic CalendarSolver functionality.
"""
import time

import numpy as np

from timealloc.calendar_solver import CalendarSolver


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
cal.optimize()
solve_time = time.time() - start_ts

# Display the results
cal.display()
array = np.reshape([y for (x,y) in cal.instance.A.get_values().items()],
                   (num_timeslots, num_tasks))
print("Schedule (timeslot x task):")
print(array)
print('Solve time', solve_time)
