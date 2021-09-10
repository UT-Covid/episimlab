import datetime as dt
import numpy as np
import pandas as pd
import xsimlab as xs


def dt64_to_day_of_week(dt64) -> int:
    """Monday == 0, Sunday == 6
    """
    assert isinstance(dt64, np.datetime64)
    index = pd.DatetimeIndex([dt64])
    return index.dayofweek[0]


def get_int_per_day(step_delta) -> float:
    """
    """
    assert isinstance(step_delta, np.timedelta64), \
        f"`step_delta` is not datetime64: {step_delta} type is {type(step_delta)}"
    return np.timedelta64(1, 'D') / step_delta


def discrete_time_approx(rate, timestep):
    """
    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """
    return 1. - (1. - rate)**(1. / timestep)


@xs.process
class IntPerDay:
    """Provide an interval per day `int_per_day`"""
    int_per_day = xs.variable(global_name='int_per_day', intent='out')

    @xs.runtime(args='step_delta')
    def run_step(self, step_delta):
        self.int_per_day = get_int_per_day(step_delta=step_delta)
