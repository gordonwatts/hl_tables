from typing import Any, List
import copy

from dataframe_expressions import DataFrame

from .runner import runner, result
from .servicex.xaod_runner import xaod_runner
from .awkward.awkward_runner import awkward_runner

runners: List[runner] = [xaod_runner(), awkward_runner()]


def make_local(df: DataFrame) -> Any:
    '''
    Get the data from the remote system that is represented by `df` and get it here, locally, on
    this computer.
    '''
    modified_df = df
    for r in runners:
        modified_df = r.process(modified_df)
        if isinstance(modified_df, result):
            break

    if not isinstance(modified_df, result):
        raise Exception('Unable to process data frame!')

    return modified_df.result
