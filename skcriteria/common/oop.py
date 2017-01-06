#~ #!/usr/bin/env python
#~ # -*- coding: utf-8 -*-

#~ import abc

#~ import numpy as np

#~ import six


#~ class InsuficientValues(ValueError):
    #~ pass


#~ @six.add_metaclass(abc.ABCMeta)
#~ class Solver(object):

    #~ def __init__(self, criterias):
        #~ self._rank = None
        #~ self._n_criteria = n_criteria
        #~ self._mtx = np.array([], dtype=np.float128)
        #~ self._weights = np.array([], dtype=np.float128)
        #~ self._criteria = np.empty(dtype=np.int8)
        #~ self._criteria_names = np.zeros(criteria, dtype=np.unicode)

    #~ @abc.abstractmethod
    #~ def solve(self):
        #~ raise  NotImplementedError()

    #~ def add_criteria(self, name, maxmin, default=None, weight=None):
        #~ self._criteria_names = np.append(self._criteria_name, name)
        #~ self._criteria = np.append(self._criteria, maxmin)
        #~ self_weights = np.append(self._weights, weight or 1)
        #~ return self

    #~ def criterias(self, *args):
        #~ for data in args:
            #~ while len(data) < 4: list(data) + [None]
            #~ self.add_criteria(*data)
        #~ return self

    #~ def add_alernative(name, *values):
        #~ if len(values) != len(self._criteria):
            #~ raise InsuficientValues("



    #~ @property
    #~ def values(self):
        #~ return self._mtx

