from dataframe_expressions import DataFrame
from typing import Tuple
import hep_tables
import hl_tables.local as local
import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np


def histogram(df: DataFrame, bins: int = 10, range: Tuple[float, float] = (0, 1)):
    hist_data = hep_tables.histogram(df, bins=bins, range=range)
    h, bins = local.make_local(hist_data)
    f, ax = plt.subplots()
    ax.step(bins, np.r_[h, h[-1]], where='post')
    # hep.histplot(h, bins)
