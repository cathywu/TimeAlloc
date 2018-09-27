from bs4 import NavigableString
import json

import ipdb
import re

import timealloc.util as util

DAYOFWEEK = re.compile('[A-Z][a-z]*')

TIME_AVAILABLE = 168
FILTERS_WHEN = ["(first thing)", "(last thing)"]
FILTERS_DAYS = {"daily, ([\.\d]+)": ['M', 'T', 'W', 'R', 'F', 'Sa', 'Su'],
                "^[A-Z][A-Za-z]*, ([\.\d]+)": DAYOFWEEK,
                "^[A-Z][A-Za-z]*$": DAYOFWEEK, }
FILTER_TASK_HOUR = "^(.*) \[([\.\d]+)( )?(hours|hrs|hour|hr)?\].*"
FILTER_TASK_MIN = "^(.*) \[([\.\d]+)( )?(min|minutes|mi)?\].*"


class TaskParser:
    def __init__(self, time_alloc_fname, tasks_fname):
        # For time allocation
        self.time_alloc = {}
        self.tags = {}
        self.running_total = 0

        # For tasks
        self.tasks = {}
        self.running_total_tasks = 0

        self.time_allocation_fname = time_alloc_fname
        t_alloc_soup = util.html_from_md(self.time_allocation_fname)
        self._time_alloc_from_soup(t_alloc_soup)

        self.tasks_fname = tasks_fname
        tasks_soup = util.html_from_md(self.tasks_fname)
        self._tasks_from_soup(tasks_soup)

    def _time_alloc_from_soup(self, soup):
        # TODO(cathywu) change this to find the "Time Alloc" heading and then
        # look at its next list
        for top in soup.find_all("ul")[0].children:
            if not isinstance(top, NavigableString):
                # print('Tag:', top.next)
                tag = top.next
                if tag not in self.tags:
                    self.tags[tag] = {}
                    self.tags[tag]['total'] = 0

                for level2 in top.next.next.children:
                    if not isinstance(level2, NavigableString):
                        # print('Category:', level2.next)
                        m = re.search('(.+) \[([\.\d]+)\].*', level2.next)
                        category = m.group(1)
                        total = float(m.group(2))

                        if category not in self.time_alloc:
                            self.time_alloc[category] = {}
                        self.time_alloc[category]['tag'] = tag
                        self.time_alloc[category]['total'] = total
                        self.tags[tag]['total'] += total
                        self.running_total += total
                    if isinstance(level2.next.next, NavigableString):
                        # print('Extra:', level2.next.next)
                        continue
                    for level3 in level2.next.next.children:
                        if not isinstance(level3, NavigableString):
                            # print('Metadata:', level3.next)
                            label, metadata = level3.next.split(": ", 1)
                            label = label.lower()
                            metadatum = metadata.split("; ")
                            if label == "when":
                                self.time_alloc[category][label] = {}
                                self.time_alloc[category][label][
                                    'constraints'] = []
                                for datum in metadatum:
                                    ignore = False
                                    for filt, daysofweek in FILTERS_DAYS.items():
                                        m = re.search(filt, datum)
                                        print("DATUM", datum)
                                        if m is not None:
                                            print(m.groups())
                                            if isinstance(daysofweek,
                                                          type(DAYOFWEEK)):
                                                self.time_alloc[category][
                                                    label][
                                                    "DaysOfWeek"] = daysofweek.findall(
                                                    datum)
                                                if len(m.groups()) > 0:
                                                    self.time_alloc[category][
                                                        label][
                                                        "HoursPerDay"] = float(
                                                        m.group(1))
                                            else:
                                                self.time_alloc[category][
                                                    label][
                                                    "DaysOfWeek"] = daysofweek
                                                if len(m.groups()) > 0:
                                                    self.time_alloc[category][
                                                        label][
                                                        "HoursPerDay"] = float(
                                                        m.group(1))
                                            ignore = True
                                    if not ignore:
                                        self.time_alloc[category][label][
                                            'constraints'].append(datum)

                            else:
                                self.time_alloc[category][label] = metadatum

                        if isinstance(level3.next.next, NavigableString):
                            # print('Extra:', level2.next.next)
                            continue
                        for e in level3.children:
                            print('Extra children (not supported):', e)

        print(json.dumps(self.time_alloc, indent=4))
        print(json.dumps(self.tags, indent=4))
        if self.running_total != TIME_AVAILABLE:
            print("WARNING: time allocation is off ({} != {} hours)".format(
                self.running_total, TIME_AVAILABLE))

    def _tasks_from_soup(self, soup):
        # FIXME(cathywu) don't hardcode this?
        category = "Work"

        # TODO(cathywu) change this to find the "Time Alloc" heading and then look at its next list
        for top in soup.find_all("ul")[0].children:
            if not isinstance(top, NavigableString):
                # print('Tag:', top.next)
                task = top.next
                # if tag not in tags:
                # tags[tag] = {}
                # tags[tag]['total'] = 0
                print(task)
                # ipdb.set_trace()
                if task in ["", "\n"]:
                    continue
                try:
                    # Parse out estimated hours for the task
                    m = re.search(FILTER_TASK_HOUR, task)
                    task = m.group(1)
                    total = float(m.group(2))
                except AttributeError:
                    # Parse out estimated minutes for the task
                    m = re.search(FILTER_TASK_MIN, task)
                    task = m.group(1)
                    total = float(m.group(2))/60  # convert to hours

                if task not in self.tasks:
                    self.tasks[task] = {}
                self.tasks[task]['total'] = total
                self.running_total_tasks += total

                if isinstance(top.next.next, NavigableString):
                    continue
                for level3 in top.next.next.children:
                    if not isinstance(level3, NavigableString):
                        # print('Metadata:', level3.next)
                        label, metadata = level3.next.split(": ", 1)
                        label = label.lower()
                        if label == "when":
                            metadatum = metadata.split("; ")
                            self.tasks[task][label] = {}
                            self.tasks[task][label]['constraints'] = []
                            for datum in metadatum:
                                ignore = False
                                for filt, daysofweek in FILTERS_DAYS.items():
                                    m = re.search(filt, datum)
                                    print("DATUM", datum)
                                    if m is not None:
                                        print(m.groups())
                                        if isinstance(daysofweek,
                                                      type(DAYOFWEEK)):
                                            self.time_alloc[category][label][
                                                "DaysOfWeek"] = daysofweek.findall(
                                                datum)
                                            if len(m.groups()) > 0:
                                                self.time_alloc[category][
                                                    label][
                                                    "HoursPerDay"] = float(
                                                    m.group(1))
                                        else:
                                            self.time_alloc[category][label][
                                                "DaysOfWeek"] = daysofweek
                                            if len(m.groups()) > 0:
                                                self.time_alloc[category][
                                                    label][
                                                    "HoursPerDay"] = float(
                                                    m.group(1))
                                        ignore = True
                                if not ignore:
                                    self.tasks[task][label][
                                        'constraints'].append(datum)

                        else:
                            self.tasks[task][label] = metadata

                    if isinstance(level3.next.next, NavigableString):
                        # print('Extra:', level2.next.next)
                        continue
                    for e in level3.children:
                        print('Extra children (not supported):', e)
        print(json.dumps(self.tasks, indent=4))
        if self.running_total_tasks != self.time_alloc['Work']['total']:
            print("WARNING: work allocation is off ({} != {} hours)".format(
                self.running_total_tasks, self.time_alloc['Work']['total']))
