from typing import Tuple
import ast

from dataframe_expressions import DataFrame
from make_it_sync import make_sync
import matplotlib.pyplot as plt
import numpy as np
from hep_tables.utils import to_ast
from dataframe_expressions import ast_DataFrame

import hl_tables.local as local


def histogram_fill(df: DataFrame,
                   bins: int = 10,
                   range: Tuple[float, float] = (0, 1),
                   density: bool = None):
    'Add a histogram to the compute graph'
    keywords = [
        ast.keyword(arg='bins', value=to_ast(bins)),
        ast.keyword(arg='range', value=to_ast(range)),
        ast.keyword(arg="density", value=to_ast(density)),
    ]

    call_node = ast.Call(func=ast.Attribute(value=ast_DataFrame(df), attr='histogram'),
                         args=[], keywords=keywords)
    return DataFrame(call_node)


async def histogram_async(df: DataFrame,
                          bins: int = 10,
                          range: Tuple[float, float] = (0, 1),
                          density: bool = None):
    '''
    Create and plot a histogram. Meant for use inside a Jupyter notebook.

    Arguments

        df              DataFrame the represents the data to be plotted (single column)
        bins            How many bins should be created
        range           Lower and upper bound of the histogram
        density         If True then normalize the area to one.

    ## Notes

    - Both async and sync versions available.

    '''
    # Add to the comput graph
    hist_data = histogram_fill(df, bins, range, density)

    # Now render locally so we can plot it.
    h, bins = await local.make_local_async(hist_data)
    f, ax = plt.subplots()
    ax.fill_between(bins, np.r_[h, h[-1]], step='post')

histogram = make_sync(histogram_async)
