"""
Utility functions for time, hour, ip slot conversions.
"""
from datetime import datetime
import time

import numpy as np

SLOTS_PER_HOUR = 2  # each slot represents 15 minutes
WEEKDAYS = {'SATURDAY': 0, 'SUNDAY': 1, 'MONDAY': 2, 'TUESDAY': 3,
            'WEDNESDAY': 4, 'THURSDAY': 5, 'FRIDAY': 6, }


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

    raise(NotImplementedError, "{} could not be read as datetime".format(s))


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

    raise (NotImplementedError, "{} could not be read as struct_time".format(s))


def struct_time_to_slot_mask(time, modifier="before", duration=None):
    """
    Maps before/after/at XX:YYam/pm to IP slot mask

    :param time:
    :param modifier:
    :param duration: before, after, or at/on
    :return:
    """
    mask = np.zeros(24*7*SLOTS_PER_HOUR)
    day_starts = [24*SLOTS_PER_HOUR*i for i in range(7)]
    day_starts.append(mask.size)  # for "after" termination edge case

    offset = hour_to_ip_slot(time.tm_hour + time.tm_min/60)
    if modifier == "before":
        for i in range(7):
            mask[day_starts[i]:day_starts[i]+offset] = 1
    elif modifier == "after":
        for i in range(7):
            mask[day_starts[i]+offset:day_starts[i+1]] = 1
    elif modifier == "on" or modifier == "at":
        if duration is None:
            raise (ValueError, "Duration not provided")
        for i in range(7):
            mask[day_starts[i]+offset:day_starts[i]+offset+duration] = 1
    else:
        raise (
        NotImplementedError, "Modifier {} not supported".format(modifier))
    return mask


def date_to_slot_mask(time, modifier="before", duration=None):
    pass


def datetime_to_slot_mask(time, modifier="before", duration=None):
    """
    Maps before/after/on weekday time to IP slot mask

    :param time:
    :param modifier: before, after, on, at
    :param duration:
    :return:
    """
    mask = np.zeros(24*7*SLOTS_PER_HOUR)
    day_starts = [24*SLOTS_PER_HOUR*i for i in range(7)]
    day_starts.append(mask.size)  # for "after" termination edge case

    weekday = (time.weekday() + 2) % 7  # Saturday is the week start
    offset = hour_to_ip_slot(time.hour + time.minute / 60)
    if modifier == "before":
        mask[:day_starts[weekday] + offset] = 1
    elif modifier == "after":
        mask[day_starts[weekday] + offset:] = 1
    elif modifier == "on":
        mask[day_starts[weekday] + offset:day_starts[weekday + 1]] = 1
    elif modifier == "at":
        if duration is None:
            raise (ValueError, "Duration not provided")
        mask[day_starts[weekday] + offset:day_starts[
                                              weekday] + offset + duration] = 1
    else:
        raise (
            NotImplementedError, "Modifier {} not supported".format(modifier))
    return mask
