from typing import Any

from dataframe_expressions import DataFrame
import hep_tables


def make_local(df: DataFrame) -> Any:
    '''
    Get the data from the remote system that is represented by `df` and get it here, locally, on
    this computer.
    '''
    return hep_tables.make_local(df)
