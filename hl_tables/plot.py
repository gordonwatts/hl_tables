from typing import Tuple

from dataframe_expressions import DataFrame
import hep_tables
from make_it_sync import make_sync
import matplotlib.pyplot as plt
import numpy as np

import hl_tables.local as local


async def histogram_async(df: DataFrame, bins: int = 10, range: Tuple[float, float] = (0, 1)):
    hist_data = hep_tables.histogram(df, bins=bins, range=range)
    h, bins = await local.make_local_async(hist_data)
    f, ax = plt.subplots()
    ax.fill_between(bins, np.r_[h, h[-1]], step='post')

histogram = make_sync(histogram_async)
