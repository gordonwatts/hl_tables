from dataframe_expressions import DataFrame
from typing import Tuple
import hep_tables
import hl_tables.local as local
import matplotlib.pyplot as plt
import numpy as np


async def histogram(df: DataFrame, bins: int = 10, range: Tuple[float, float] = (0, 1)):
    hist_data = hep_tables.histogram(df, bins=bins, range=range)
    h, bins = await local.make_local_async(hist_data)
    f, ax = plt.subplots()
    ax.fill_between(bins, np.r_[h, h[-1]], step='post')
