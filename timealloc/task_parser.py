from bs4 import NavigableString
import json

import re

import timealloc.util as util

DAYOFWEEK = re.compile('[A-Z][a-z]*')

TIME_AVAILABLE = 14.5 * 7 # 18 * 7  # 168
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

        # For other tasks
        self.other_tasks = {}
        self.running_total_other_tasks = 0

        self.time_allocation_fname = time_alloc_fname
        t_alloc_soup = util.html_from_md(self.time_allocation_fname)
        self._time_alloc_from_soup(t_alloc_soup)

        self.tasks_fname = tasks_fname
        tasks_soup = util.html_from_md(self.tasks_fname)

        # important and urgent tasks
        work_tasks0, work_tasks_total0 = self._tasks_from_soup(tasks_soup,
            heading="Work: Important and urgent", category="Work")
        work_tasks0 = self._tag_important(work_tasks0)
        work_tasks0 = self._tag_urgent(work_tasks0)
        work_tasks0 = self._tag_soon(work_tasks0)

        # important and not urgent tasks
        work_tasks1, work_tasks_total1 = self._tasks_from_soup(tasks_soup,
            heading="Work: Important and not urgent", category="Work")
        work_tasks1 = self._tag_important(work_tasks1)

        # not important and urgent tasks
        work_tasks2, work_tasks_total2 = self._tasks_from_soup(tasks_soup,
            heading="Work: Not important and urgent", category="Work")
        work_tasks2 = self._tag_urgent(work_tasks2)
        work_tasks2 = self._tag_soon(work_tasks2)

        # not important and not urgent tasks
        work_tasks3, work_tasks_total3 = self._tasks_from_soup(tasks_soup,
            heading="Work: other", category="Work")

        # merge tasks into a single set of work tasks
        self.work_tasks = self._merge_tasks((work_tasks0, work_tasks1,
                                             work_tasks2, work_tasks3))
        self.work_tasks = self._add_category(self.work_tasks, "Work")
        self.work_tasks_total = work_tasks_total0 + work_tasks_total1 + \
                                work_tasks_total2 + work_tasks_total3

        # load non-work tasks
        # load errand tasks
        errand_tasks0, errand_tasks_total0 = self._tasks_from_soup(tasks_soup,
            heading="Errand: Important and urgent", category="Errand")
        errand_tasks0 = self._tag_important(errand_tasks0)
        errand_tasks0 = self._tag_urgent(errand_tasks0)
        errand_tasks0 = self._tag_soon(errand_tasks0)

        # important and not urgent tasks
        errand_tasks1, errand_tasks_total1 = self._tasks_from_soup(tasks_soup,
            heading="Errand: Important and not urgent", category="Errand")
        errand_tasks1 = self._tag_important(errand_tasks1)

        # not important and urgent tasks
        errand_tasks2, errand_tasks_total2 = self._tasks_from_soup(tasks_soup,
            heading="Errand: Not important and urgent", category="Errand")
        errand_tasks2 = self._tag_urgent(errand_tasks2)
        errand_tasks2 = self._tag_soon(errand_tasks2)

        # not important and not urgent tasks
        errand_tasks3, errand_tasks_total3 = self._tasks_from_soup(tasks_soup,
            heading="Errand: other", category="Errand")

        # merge tasks into a single set of errand tasks
        self.errand_tasks = self._merge_tasks(
            (errand_tasks0, errand_tasks1, errand_tasks2, errand_tasks3))
        self.errand_tasks = self._add_category(self.errand_tasks, "Errand")
        self.errand_tasks_total = errand_tasks_total0 + errand_tasks_total1 + \
                                errand_tasks_total2 + errand_tasks_total3

        other_tasks, other_tasks_total = self._tasks_from_soup(
            tasks_soup, heading="Other tasks")
        persistent_tasks, persistent_tasks_total = self._tasks_from_soup(
            tasks_soup, heading="Persistent tasks")
        # merge tasks into a single set of other tasks
        self.other_tasks = self._merge_tasks((other_tasks, self.errand_tasks,
                                             persistent_tasks))

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
                        m = re.search('(.+) \[([ .,\d]+)\].*', level2.next)
                        category = m.group(1)
                        total = m.group(2).split(', ')

                        if category not in self.time_alloc:
                            self.time_alloc[category] = {}
                        self.time_alloc[category]['tag'] = tag

                        self.time_alloc[category]['total'] = float(total[0])
                        self.time_alloc[category]['min'] = float(total[0])
                        if len(total) > 1:
                            self.time_alloc[category]['max'] = float(total[1])

                        self.tags[tag]['total'] += float(total[0])
                        self.running_total += float(total[0])
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
                                if label not in self.time_alloc[category]:
                                    self.time_alloc[category][label] = []
                                self.time_alloc[category][label].append(metadata)
                            else:
                                self.time_alloc[category][label] = metadata

                        if isinstance(level3.next.next, NavigableString):
                            # print('Extra:', level2.next.next)
                            continue
                        for e in level3.children:
                            print('Extra children (not supported):', e)

        # print(json.dumps(self.time_alloc, indent=4))
        # print(json.dumps(self.tags, indent=4))
        if self.running_total != TIME_AVAILABLE:
            print("WARNING: time allocation is off ({} != {} hours)".format(
                self.running_total, TIME_AVAILABLE))

    def _tasks_from_soup(self, soup, category=None, heading="Work tasks"):
        headings = soup.find_all("h2")
        tasks_dict = {}
        running_total = 0

        h2 = [h2.next for h2 in soup.find_all("h2")].index(heading)
        tasks_ul = headings[h2].next.next.next
        for top in tasks_ul.children:
            if not isinstance(top, NavigableString):
                # print('Tag:', top.next)
                task = top.next
                # if tag not in tags:
                # tags[tag] = {}
                # tags[tag]['total'] = 0
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

                if task not in tasks_dict:
                    tasks_dict[task] = {}
                tasks_dict[task]['total'] = total
                running_total += total

                if isinstance(top.next.next, NavigableString):
                    continue
                for level3 in top.next.next.children:
                    if not isinstance(level3, NavigableString):
                        # print('Metadata:', level3.next)
                        label, metadata = level3.next.split(": ", 1)
                        label = label.lower()
                        if label == "when":
                            if label not in tasks_dict[task]:
                                tasks_dict[task][label] = []
                            tasks_dict[task][label].append(metadata)
                        elif label == "categories":
                            if label not in tasks_dict[task]:
                                tasks_dict[task][label] = []
                            categories = metadata.split(", ")
                            tasks_dict[task][label].extend(categories)
                        else:
                            tasks_dict[task][label] = metadata

                    if isinstance(level3.next.next, NavigableString):
                        # print('Extra:', level2.next.next)
                        continue
                    for e in level3.children:
                        print('Extra children (not supported):', e)
        # print(json.dumps(self.tasks, indent=4))
        if category is not None:
            if running_total != self.time_alloc[category]['total']:
                print("WARNING({}): {} allocation is off ({} != {} "
                      "hours)".format(heading, category, running_total,
                    self.time_alloc[category]['total']))

        return tasks_dict, running_total

    @staticmethod
    def _tag_important(tasks):
        for key in tasks.keys():
            tasks[key]["important"] = True
        return tasks

    @staticmethod
    def _tag_urgent(tasks):
        for key in tasks.keys():
            tasks[key]["urgent"] = True
        return tasks

    @staticmethod
    def _tag_soon(tasks):
        for key in tasks.keys():
            tasks[key]["soon"] = True
        return tasks

    @staticmethod
    def _merge_tasks(listoftasks):
        overall_tasks = {}
        for tasks in listoftasks:
            for k, v in tasks.items():
                overall_tasks[k] = v
        return overall_tasks

    @staticmethod
    def _add_category(tasks, category):
        for key in tasks.keys():
            if "categories" not in tasks[key]:
                tasks[key]["categories"] = []
            tasks[key]["categories"].append(category)
        return tasks
