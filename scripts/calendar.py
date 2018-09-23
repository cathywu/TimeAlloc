import numpy as np

from timealloc.calendar_solver import CalendarSolver


# User defined parameters
num_tasks = 5
num_timeslots = 10

utilities = np.ones((num_tasks, num_timeslots)).T
utilities[0, 0] = 0  # forces entry (0,0) to be unassigned

# Prepare the IP
params = {
    'num_tasks': num_tasks,
    'num_timeslots': num_timeslots,
}
cal = CalendarSolver(utilities, params)

# Solve and display the results
cal.optimize()
cal.display()

array = np.reshape([y for (x,y) in cal.instance.A.get_values().items()],
                   (num_timeslots, num_tasks))
print("Schedule (timeslot x task):")
print(array)
