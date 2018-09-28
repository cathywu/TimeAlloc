Input formats
-------------

Supported inputs are markdown files (`.md`).


Task list format
================

The basic format is:

::

    - [Task name]
        - [Attribute]: [Value]

Example:

::

    - Watch: Giraffes vs lions video [10 min]
        - When: after 6pm
    - Read: Giraffes are now endangered [30 min]
        - Before: Thursday
    - Research: How to save giraffes [3 hours]
        - Chunks: 1-1.5
        - Before: 3pm
        - Spread: true

Valid attributes are:

- Before: date/time
- After: date/time
- On: date/time
- At: date/time
- Chunks: hours or hours range
- Display name: name (permit alternate/short display name for Bokeh
  visualization)
- Spread: (optional; include to encourage scheduling chunks on multiple days)

Not yet supported

- Priority: 0-100
- Urgency: 0-100
- Prefer: date/time clause
- When: date/time (TBD)
- At: location
- In: location
- Deadline: date/time
- Deadline?: date/time (uncertain deadline)

Categories
==========

Privileged categories (not yet supported)

- Work
- Break
- Lunch
- Dinner
- Errands
- Sleep
- Social
- Outreach/service
- Me
- Wellbeing