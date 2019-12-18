#!/usr/bin/env python
# -*- coding: utf-8 -*-

# flake8: noqa
# pragma: no cover

# From Pandas:
#   https://github.com/pandas-dev/pandas/blob/master/pandas/core/base.py


class AccessorProperty(object):
    """Descriptor for implementing accessor properties like Series.str
    """
    def __init__(self, accessor_cls, construct_accessor):
        self.accessor_cls = accessor_cls
        self.construct_accessor = construct_accessor
        self.__doc__ = accessor_cls.__doc__

    def __get__(self, instance, owner=None):
        if instance is None:
            # this ensures that Series.str.<method> is well defined
            return self.accessor_cls
        return self.construct_accessor(instance)

    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        raise AttributeError("can't delete attribute")
