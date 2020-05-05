from dataframe_expressions import DataFrame
from typing import Tuple
import hep_tables
from hl_tables.local import make_local
import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np


def histogram(df: DataFrame, bins: int = 10, range: Tuple[float, float] = (0, 1)):
    hist_data = hep_tables.histogram(df)
    h, bins = make_local(hist_data)
    f, ax = plt.subplots()
    ax.step(bins, np.r_[h, h[-1]], step='post')
    hep.histplot(h, bins)

    # from hl_tables.local import make_local
    # dummy = make_local(None)
    # dummy = None
