"""
Script for CalendarSolver functionality with integrated markdown parser.
"""
import time

from timealloc.tasks import Tasks
from timealloc.calendar_solver import CalendarSolver

# User specified input files
# time_allocation_fname = "scratch/time-allocation-2018-09-28-simple.md"
time_allocation_fname = "scratch/time-allocation-2018-10-11.md"
tasks_fname = "scratch/tasks-2018-10-11.md"
solution_fname = "tmpt6ws14e_.cplex.sol"
lp_fname = "tmp694awfk4.pyomo.lp"

tasks = Tasks(time_allocation_fname, tasks_fname)

tasks.display()

params = tasks.get_ip_params()
cal = CalendarSolver(tasks.utilities, params)

# Solve
print("Optimizing...")
start_ts = time.time()
cal.optimize()
# cal.load(lp_fname, solution_fname)
solve_time = time.time() - start_ts

# Display the results
cal.visualize()
cal.display()
cal.get_diagnostics()
print('Solve time', solve_time)
