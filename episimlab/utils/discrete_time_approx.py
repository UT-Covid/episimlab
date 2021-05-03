from math import isclose
from ..cy_utils.cy_utils import discrete_time_approx_wrapper as cy_dta


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

    val = 1. - (1. - rate)**(1. / timestep)
    cy_val = cy_dta(rate, timestep)
    assert isclose(val, cy_val, rel_tol=1e-7), (val, cy_val)
    return val
