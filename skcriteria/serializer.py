#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to serialize ``skcriteria.data.Data`` instances.

"""


# =============================================================================
# META
# =============================================================================

__all__ = ['DataSerializerProxy']


# =============================================================================
# IMPORTS
# =============================================================================

from tabulate import tabulate

from .attribute import AttributeClass


# =============================================================================
# CONSTANTS
# =============================================================================

FLOAT_FMT = ".5f"

WEIGHT_FMT = FLOAT_FMT

MTX_FMT = FLOAT_FMT

TABULATE_PARAMS = {
    "tablefmt": "simple",
    "headers": "firstrow",
    "numalign": "center",
    "stralign": "center",
}


# =============================================================================
# SERIALIZER CLASS
# =============================================================================

class DataSerializerProxy(AttributeClass):

    data = AttributeClass.parameter()

    weight_fmt_ = AttributeClass.parameter(init=False, default=WEIGHT_FMT)
    mtx_fmt_ = AttributeClass.parameter(init=False, default=MTX_FMT)
    tbl_params_ = AttributeClass.parameter(init=False, default=TABULATE_PARAMS)

    __configuration__ = {
        "repr": False, "frozen": True,
        "order": False, "eq": False}

    @data.validator
    def _data_validator(self, attr, value):
        from .data import Data
        if not isinstance(value, Data):
            raise TypeError(
                f"'data' must be an instance of 'skcriteria.data.Data'")

    def to_string_table(self):
        from .data import CRITERIA_STR

        data = self.data

        # header
        header = ["A/C"]
        if data.weights is None:
            header_it = zip(data.cnames, data.criteria)
            for cname, sense in header_it:
                str_sense = CRITERIA_STR[sense]
                header.append(
                    f"{cname} ({str_sense})")
        else:
            header_it = zip(data.cnames, data.criteria, data.weights)
            for cname, sense, weight in header_it:
                str_sense = CRITERIA_STR[sense]
                str_weight = format(weight, self.weight_fmt_)
                header.append(
                    f"{cname} ({str_sense}) W={str_weight}")

        # body
        body = []
        body_it = zip(data.anames, data.mtx)
        for aname, mrow in body_it:
            str_mrow = list(
                map(lambda e: format(e, self.mtx_fmt_), mrow))
            body.append([aname] + str_mrow)

        return [header] + body

    def to_text(self):
        str_table = self.to_string_table()
        return tabulate(str_table, **self.tbl_params_)

    def to_html(self):
        str_table = self.to_string_table()

        params = dict(self.tbl_params_)
        params["tablefmt"] = "html"

        return tabulate(str_table, **params)
