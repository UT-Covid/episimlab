import yaml
import string
import xarray as xr
import xsimlab as xs
import pandas as pd
import numpy as np
from itertools import product

from .toy import SetupPhiWithToyPartitioning


@xs.process
class SetupPhiWithPartitioning(SetupPhiWithToyPartitioning):
    """Uses xarray, not pandas
    """
    pass
