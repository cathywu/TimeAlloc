"""
Utility functions for time, hour, ip slot conversions.
"""
from datetime import datetime
import time

import numpy as np

# TODO(cathywu) move to config
LOOKAHEAD = 7  # planning horizon is 7 days
DAY_START = 7  # first hour of the day is 7am
SLOTS_PER_HOUR = 2  # each slot represents 30 minutes
HOURS_PER_DAY = 14.5  # number of hours valid for scheduling
# includes a dummy end variable to mark the end of the day
SLOTS_PER_DAY = int(HOURS_PER_DAY * SLOTS_PER_HOUR + 1)
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY
NUMSLOTS = LOOKAHEAD * SLOTS_PER_DAY
WEEKDAYS = {
    'SATURDAY': 0,
    'SUNDAY': 1,
    'MONDAY': 2,
    'TUESDAY': 3,
    'WEDNESDAY': 4,
    'THURSDAY': 5,
    'FRIDAY': 6,
    'Sa': 0,
    'Su': 1,
    'M': 2,
    'T': 3,
    'W': 4,
    'R': 5,
    'F': 6,
}
MODIFIERS = ['after', 'before', 'at', 'on']


def hour_to_ip_slot(hour):
    return int(hour * SLOTS_PER_HOUR)


def ip_slot_to_hour(slot):
    return float(slot) / SLOTS_PER_HOUR


def text_to_datetime(str, weekno, year):
    """
    Week number and year are needed for the resulting datetime object to have
    the correct day of the week. Otherwise, it will assume year 1990, and use
    the corresponding day of the week. Week number is redundant when the
    date is (optionally) provided.

    :param str: String following one of the patterns below, minus the weekno
    and year
    :param weekno: Week number of the year (0-51)
    :param year: Year (e.g. 2018)
    :return: datetime object
    """
    patterns = [
        "%A %m/%d %I:%M%p %W %Y",  # Tuesday 9/7 1:30pm 36 2018
        "%a %m/%d %I:%M%p %W %Y",  # Tues 9/7 1:30pm 36 2018
        "%A %m/%d %I%p %W %Y",     # Tuesday 9/7 1pm 36 2018
        "%a %m/%d %I%p %W %Y",     # Tues 9/7 1pm 36 2018
        "%A %I:%M%p %W %Y",        # Tuesday 1:30pm 36 2018
        "%a %I:%M%p %W %Y",        # Tues 1:30pm 36 2018
        "%A %I%p %W %Y",           # Tuesday 1pm 36 2018
        "%a %I%p %W %Y",           # Tues 1pm 36 2018
        "%A %m/%d %W %Y",          # Tuesday 9/7 36 2018
        "%a %m/%d %W %Y",          # Tues 9/7 36 2018
        "%A %W %Y",                # Tuesday 36 2018
        "%a %W %Y",                # Tues 36 2018
    ]

    for i in range(len(patterns)):
        try:
            s = datetime.strptime("{} {} {}".format(str, weekno, year),
                                  patterns[i])
            return s
        except ValueError:
            pass

    raise ValueError("{} could not be read as datetime".format(str))


def text_to_struct_time(str):
    """
    Format found here: https://docs.python.org/2/library/time.html

    :param str:
    :return:
    """
    patterns = [
        "%I:%M%p",  # 1:30pm
        "%I:%M%p",  # 1:30pm
        "%I%p",     # 1pm
        "%I%p",     # 1pm
    ]

    for i in range(len(patterns)):
        try:
            s = time.strptime(str, patterns[i])
            return s
        except ValueError:
            pass

    raise ValueError("{} could not be read as struct_time".format(str))


def struct_time_to_slot_mask(time, modifier="before", duration=None):
    """
    Maps before/after/at XX:YYam/pm to IP slot mask. Applies the mask to each
    day in the lookahead time slot mask.

    :param time:
    :param modifier:
    :param duration: before, after, or at/on
    :return:
    """
    mask = np.zeros(NUMSLOTS)
    day_starts = [SLOTS_PER_DAY*i for i in range(LOOKAHEAD)]
    day_starts.append(mask.size)  # for "after" termination edge case

    offset = hour_to_ip_slot(time.tm_hour + time.tm_min/60 - DAY_START)
    if modifier == "before":
        for i in range(LOOKAHEAD):
            mask[day_starts[i]:day_starts[i]+offset] = 1
    elif modifier == "after":
        for i in range(LOOKAHEAD):
            mask[day_starts[i]+offset:day_starts[i+1]] = 1
    elif modifier == "on" or modifier == "at":
        if duration is None:
            raise (ValueError, "Duration not provided")
        for i in range(LOOKAHEAD):
            mask[day_starts[i]+offset:day_starts[i]+offset+duration] = 1
    else:
        raise ValueError("Modifier {} not supported".format(modifier))
    return mask


def date_to_slot_mask(time, modifier="before", duration=None):
    pass


def datetime_to_slot_mask(time, modifier="before", start=None, duration=None):
    """
    Maps before/after/on weekday time to IP slot mask

    :param time:
    :param modifier: before, after, on, at
    :param duration:
    :return:
    """
    mask = np.zeros(NUMSLOTS)
    day_starts = [SLOTS_PER_DAY*i for i in range(LOOKAHEAD)]
    day_starts.append(mask.size)  # for "after" termination edge case

    if start is None:
        day = (time.weekday() + 2) % 7  # Saturday is the week start
    else:
        day = (time-start).days
        if day > LOOKAHEAD or day < 0:
            raise IOError("Cannot support event {} days ahead".format(day))

    if time.hour != 0 or time.minute != 0:
        # if a time is provided, then shift it by DAY_START
        offset = hour_to_ip_slot(time.hour + time.minute / 60 - DAY_START)
    else:
        # otherwise offset should be 0
        offset = hour_to_ip_slot(time.hour + time.minute / 60)

    # FIXME(cathywu) this is a hack
    if day == 7:
        day = 0

    if modifier == "before":
        mask[:day_starts[day] + offset] = 1
    elif modifier == "after":
        mask[day_starts[day] + offset:] = 1
    elif modifier == "on":
        mask[day_starts[day] + offset:day_starts[day + 1]] = 1
    elif modifier == "at":
        if duration is None:
            raise IOError("Duration not provided")
        mask[day_starts[day] + offset:day_starts[day] + offset + duration] = 1
    else:
        raise (
            NotImplementedError, "Modifier {} not supported".format(modifier))
    return mask


def modifier_mask(clause, start=None, total=0, weekno=None, year=None):
    sub_mask = np.zeros(SLOTS_PER_WEEK)
    subclauses = clause.split('; ')
    for subclause in subclauses:
        modifier, attribute = subclause.split(' ', 1)
        attributes = attribute.split(', ')
        for attr in attributes:
            # FIXME(cathywu) the following needs refactoring. It's a mess and
            # error-prone.
            # print(task, key, attr)
            try:
                stime = text_to_struct_time(attr)
                mask = struct_time_to_slot_mask(stime, modifier=modifier,
                                                duration=hour_to_ip_slot(total))
            except ValueError:
                try:
                    dtime = text_to_datetime(attr, weekno=weekno, year=year)
                    mask = datetime_to_slot_mask(dtime, modifier=modifier,
                                                 start=start,
                                                 duration=hour_to_ip_slot(
                                                     total))
                except (IOError, OSError, TypeError, ValueError):
                    try:
                        # Use the next week if the selected datetime actually
                        # corresponds to the past.
                        # TODO(cathywu) There should be a better solution to
                        # this
                        dtime = text_to_datetime(attr, weekno=weekno + 1,
                                                 year=year)
                        mask = datetime_to_slot_mask(dtime, modifier=modifier,
                                                     start=start,
                                                     duration=hour_to_ip_slot(
                                                         total))
                    except IOError:
                        raise IOError("{} {} not supported".format(modifier,
                                                                   attr))
            sub_mask = np.logical_or(sub_mask, mask)
    return sub_mask


def parse_days(string):
    """
    Parses strings of the form "M Sa Su R F; 4" into a size-7 binary map and
    a number of required days.
    Special strings include "daily".

    :param string:
    :return:
    """
    if string == "daily":
        return np.ones(7), 7
    else:
        mask = np.zeros(7)
        string_bits = string.split('; ', 1)
        days = string_bits[0].split(" ")
        if len(string_bits) > 1:
            total = int(string_bits[-1])
        else:
            total = len(days)
        for day in days:
            mask[WEEKDAYS[day]] = 1
        return mask, total

