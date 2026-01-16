"""
Time handling utility module.

This module provides time format conversion and time-series generation
functions, mainly for hydrological data processing. It supports
conversions between multiple time formats and array-like handling of
time ranges.

Main functionalities:
- time format conversion between integer dates, `datetime.date`, and
  `datetime.datetime`
- time-series generation based on a given range and step
- standardized time handling interfaces for CAMELS-style datasets and
  hydrological models
"""

import datetime as dt
import numpy as np


def t2dt(t, hr=False):
    """
    Convert various time formats to a standard date or datetime.

    This function is used to unify time representations from different
    sources so that subsequent processing is consistent.

    Args:
        t: input time, one of:
           - int: integer date in the form YYYYMMDD, e.g. 20200101
           - datetime.date: Python date object
           - datetime.datetime: Python datetime object
        hr (bool): whether to return a datetime with hour information
                  - False: return a `date` object (default)
                  - True: return a `datetime` object

    Returns:
        datetime.date or datetime.datetime: converted time object

    Raises:
        Exception: if the input format is not recognized

    Examples:
        >>> t2dt(20200101)                 # datetime.date(2020, 1, 1)
        >>> t2dt(20200101, hr=True)       # datetime.datetime(2020, 1, 1, 0, 0)
        >>> t2dt(dt.date(2020, 1, 1))     # datetime.date(2020, 1, 1)
    """
    tOut = None

    # integer date format (YYYYMMDD)
    if type(t) is int:
        # 检查是否为有效的8位日期格式 (10000000 < date < 30000000)
        if t < 30000000 and t > 10000000:
            # parse integer as string date
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            # choose output type based on `hr`
            tOut = t if hr is False else t.datetime()

    # `datetime.date` input
    if type(t) is dt.date:
        # choose output type based on `hr`
        tOut = t if hr is False else t.datetime()

    # `datetime.datetime` input
    if type(t) is dt.datetime:
        # choose output type based on `hr`
        tOut = t.date() if hr is False else t

    # unrecognized input format
    if tOut is None:
        raise Exception('hydroMLlib.utils.t2dt failed')
    return tOut


def tRange2Array(tRange, *, step=np.timedelta64(1, 'D')):
    """
    Convert a time range into a numpy datetime array.

    Generate an evenly spaced time-series array given a start and end
    time. This is mainly used to construct continuous time axes for
    hydrological models.

    Args:
        tRange (list/tuple): time range with two elements
                            [start_time, end_time]
                            - both endpoints can be any format supported
                              by `t2dt`
                            - the end time is excluded (half-open interval)
        step (numpy.timedelta64): time step, default is 1 day
                                  (np.timedelta64(1, 'D')). Can be
                                  changed to hours, minutes, etc.

    Returns:
        numpy.ndarray: time-series array from start to end with the
                      specified step, dtype is `numpy.datetime64`;
                      end time is excluded.

    Examples:
        >>> tRange2Array([20200101, 20200105])  
        # array(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'], dtype='datetime64[D]')
        >>> tRange2Array([20200101, 20200105], step=np.timedelta64(2, 'D'))
        # array(['2020-01-01', '2020-01-03'], dtype='datetime64[D]')

    Notes:
        - This function calls `t2dt` internally to normalize input times.
        - The generated time array can be used as the time axis for
          CAMELS-style datasets.
        - The `step` argument must be of type `numpy.timedelta64`.
    """
    # convert endpoints to standard date format
    sd = t2dt(tRange[0])  # start date
    ed = t2dt(tRange[1])  # end date

    # generate the datetime64 array from start to end (end excluded)
    tArray = np.arange(sd, ed, step)
    return tArray




