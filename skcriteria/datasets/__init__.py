#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The :mod:`skcriteria.datasets` module includes utilities to load \
datasets."""


# =============================================================================
# IMPORRTS
# =============================================================================

import json
import os
import pathlib

from skcriteria.core.data import mkdm

from .. import core

# =============================================================================
# CONSTANTS
# =============================================================================

_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_simple_stock_selection():
    """Simple stock selection decision matrix.

    This matrix was designed primarily for teaching and evaluating the behavior
    of an experiment.

    Among the data we can find: two maximization criteria (ROE, CAP),
    one minimization criterion (RI), dominated alternatives (FX), and
    one alternative with an outlier criterion (ROE, MM = 1).

    Although the criteria and alternatives are original from the authors of
    Scikit-Criteria, the numerical values were extracted at some point from a
    somewhere which we have forgotten.

    Description:

    In order to decide to buy a series of stocks, a company studied 5 candidate
    investments: PE, JN, AA, FX, MM and GN. The finance department decides to
    consider the following criteria for selection:

        1. ROE (Max): Return % for each monetary unit invested.
        2. CAP (Max): Years of market capitalization.
        3. RI (Min): Risk of the stock.

    """
    dm = core.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [3, 4, 36],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "FX", "MM", "GN"],
        criteria=["ROE", "CAP", "RI"],
    )
    return dm


def load_van2021evaluation(windows_size=7):
    r"""Dataset extracted from from historical time series cryptocurrencies.

    This dataset is extracted from::

        Van Heerden, N., Cabral, J. y Luczywo, N. (2021). Evaluación de la
        importancia de criterios para la selección de criptomonedas.
        XXXIV ENDIO - XXXII EPIO Virtual 2021, Argentina.

    The nine available alternatives are based on the ranking of the 20
    cryptocurrencies with the largest market capitalization calculated on the
    basis of circulating supply, according to information retrieved from
    Cryptocurrency Historical Prices" retrieved on July 21st, 2021, from
    there only the  coins with  complete data between October 9th, 2018 to July
    6th of 2021, excluding stable-coins, since they maintain a stable price and
    therefore do not  carry associated yields; the alternatives that met these
    requirements  turned out to be: Cardano (ADA), Binance coin (BNB),
    Bitcoin (BTC),  Dogecoin (DOGE), Ethereum (ETH), Chainlink (LINK),
    Litecoin (LTC),  Stellar (XLM) and Ripple (XRP).

    Two decision matrices were created for two sizes of overlapping moving
    windows: 7 and 15 days. Six criteria were defined on these windows that
    seek to represent returns and risks:

    - ``xRv`` - average Window return (:math:`\bar{x}RV`) - Maximize: is the
      average of the differences between the closing price of the
      cryptocurrency on the last day and the first day of each window, divided
      by the price on the first day.
    - ``sRV`` - window return deviation (:math:`sRV`) - Minimize: is the
      standard deviation of window return. The greater the deviation, the
      returns within the windows have higher variance and are unstable.
    - ``xVV`` - average of the volume of the window (:math:`\bar{x}VV`) -
      Maximize: it is the average of the summations of the transaction amount
      of the cryptocurrency in dollars in each window, representing a liquidity
      measure of the asset.
    - ``sVV`` - window volume deviation (:math:`sVV`) - Minimize: it is the
      deviation of the window volumes. The greater the deviation, the volumes
      within the windows have higher variance and are unstable.
    - ``xR2`` - mean of the correlation coefficient (:math:`\bar{x}R^2`) -
      Maximize: it is the mean of the :math:`R^2` of the fit of the linear
      trends with respect to the data. It is a measure that defines how well it
      explains that linear trend to the data within the window.
    - ``xm`` - mean of the slope (:math:`\bar{x}m`) - Maximize: it is the mean
      of the slope of the linear trend between the closing prices in dollars
      and the volumes traded in dollars of the cryptocurrency within each
      window.

    Parameters
    ----------
    windows_size: 7 o 15, default 7
        If the decision matrix based on 7 or 15 day overlapping moving windows
        is desired.


    References
    ----------
    :cite:p:`van2021evaluation`
    :cite:p:`van2021epio_evaluation`
    :cite:p:`rajkumar_2021`

    """
    paths = {
        7: _PATH / "van2021evaluation" / "windows_size_7.json",
        15: _PATH / "van2021evaluation" / "windows_size_15.json",
    }

    path = paths.get(windows_size)
    if path is None:
        raise ValueError(
            f"Windows size must be '7' or '15'. Found {windows_size!r}"
        )

    with open(path) as fp:
        data = json.load(fp)

    return mkdm(**data)
