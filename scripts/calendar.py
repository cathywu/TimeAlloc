"""
Script for basic CalendarSolver functionality.
"""
import time

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.palettes import d3
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
import numpy as np

from timealloc.calendar_solver import CalendarSolver

COLORS = d3['Category20'][20]


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
overall_mask = np.ones((num_tasks, num_timeslots)).T

# Prepare the IP
params = {
    'num_tasks': num_tasks,
    'num_timeslots': num_timeslots,
    'task_duration': task_duration,
    'task_chunk_min': task_chunk_min,
    'task_chunk_max': task_chunk_max,
    'task_valid': overall_mask,
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

# Visualize the calendar
x, y = array.nonzero()
top = x % 6
bottom = top-0.95
left = np.floor(x/6)
right = left+0.95
desc = [str(k)*15 for k in y]
print('x', x)
print('y', y)
print('t', top)
print('l', left)
print('d', desc)

colors = [COLORS[i] for i in y]
source = ColumnDataSource(data=dict(
    top=top,
    bottom=bottom,
    left=left,
    right=right,
    desc=y,
    colors=colors,
))

TOOLTIPS = [
    ("index", "$index"),
    # ("(x,y)", "($x, $y)"),
    ("(t,l)", "(@top, @left)"),
    # ("fill color", "$color[hex, swatch]:fill_color"),
    ("desc", "@desc"),
]

p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
           title="Calendar")
output_file("calendar.html")
p.xaxis[0].axis_label = 'Weekday (Sun-Fri)'
p.yaxis[0].axis_label = 'Hour (12AM-12AM)'

p.quad(top='top', bottom='bottom', left='left',
       right='right', color='colors', source=source)

source2 = ColumnDataSource(data=dict(
    x=left,
    y=top,
    names=desc,
))

labels = LabelSet(x='x', y='y', text='names', level='glyph',
              x_offset=5, y_offset=-23, source=source2, render_mode='canvas')

p.add_layout(labels)

show(p)
