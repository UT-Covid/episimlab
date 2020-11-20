import numpy as np


def discrete_time_approx(rate, timestep):
    """
    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """
    # if rate >= 1:
        # return np.nan
    # elif timestep == 0:
        # return np.nan
    return (1. - (1. - rate)**(1/timestep))
