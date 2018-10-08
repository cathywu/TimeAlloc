"""
Utility functions for time, hour, ip slot conversions.
"""
from datetime import datetime
import time

import numpy as np

# TODO(cathywu) move to config
LOOKAHEAD = 7
SLOTS_PER_HOUR = 2  # each slot represents 30 minutes
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR
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
    Format found here: https://docs.python.org/2/library/time.html
    :param str:
    :return:
    """
    patterns = [
        "%A %m/%d %I:%M%p %W %Y",  # Tuesday 9/7 1:30pm 36 2018
        "%a %m/%d %I:%M%p %W %Y",  # Tues 9/7 1:30pm 36 2018
        "%A %m/%d %I%p %W %Y",     # Tuesday 9/7 1pm 36 2018
        "%a %m/%d %I%p %W %Y",     # Tues 9/7 1pm 36 2018
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

    offset = hour_to_ip_slot(time.tm_hour + time.tm_min/60)
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
        if day > LOOKAHEAD:
            raise (ValueError, "Cannot support event {} days ahead".format(day))
    offset = hour_to_ip_slot(time.hour + time.minute / 60)
    if modifier == "before":
        mask[:day_starts[day] + offset] = 1
    elif modifier == "after":
        mask[day_starts[day] + offset:] = 1
    elif modifier == "on":
        if day == 7:
            # FIXME(cathywu) this is a hack
            day = 0
        mask[day_starts[day] + offset:day_starts[day + 1]] = 1
    elif modifier == "at":
        if duration is None:
            raise (ValueError, "Duration not provided")
        mask[day_starts[day] + offset:day_starts[
                                              day] + offset + duration] = 1
    else:
        raise (
            NotImplementedError, "Modifier {} not supported".format(modifier))
    return mask


def modifier_mask(clause, start=None, total=0, weekno=39, year=2018):
    sub_mask = np.zeros(SLOTS_PER_WEEK)
    subclauses = clause.split('; ')
    for subclause in subclauses:
        modifier, attribute = subclause.split(' ', 1)
        attributes = attribute.split(', ')
        for attr in attributes:
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
                except UnboundLocalError:
                    raise (NotImplementedError,
                           "{} {} not supported".format(modifier, attr))
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

