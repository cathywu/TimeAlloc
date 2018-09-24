import numpy as np

from timealloc.calendar_solver import CalendarSolver


# User defined parameters
num_tasks = 5
num_timeslots = 30

utilities = np.ones((num_tasks, num_timeslots)).T
utilities[0, 0] = 0  # forces entry (0,0) to be unassigned
task_chunk_min = 2*np.ones(num_tasks)
task_chunk_max = 3*np.ones(num_tasks)
task_duration = 6*np.ones(num_tasks)

# Prepare the IP
params = {
    'num_tasks': num_tasks,
    'num_timeslots': num_timeslots,
    'task_duration': task_duration,
    'task_chunk_min': task_chunk_min,
    'task_chunk_max': task_chunk_max,
}
cal = CalendarSolver(utilities, params)

# Solve and display the results
cal.optimize()
cal.display()

array = np.reshape([y for (x,y) in cal.instance.A.get_values().items()],
                   (num_timeslots, num_tasks))
print("Schedule (timeslot x task):")
print(array)
