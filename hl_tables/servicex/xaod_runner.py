from typing import Union, Tuple

from dataframe_expressions import DataFrame

from ..runner import result, runner
import hep_tables


class xaod_runner(runner):
    '''
    We can do a xaod on servicex
    '''

    def _process_by_depth(self, df: DataFrame) -> Tuple[bool, Union[DataFrame, result]]:
        'Process by depth'

        # Lets do the really easy thing first - what happens when we are at the bottom.
        if df.parent is None:
            assert df.child_expr is None, 'Internal Programming Error'
            return (isinstance(df, hep_tables.xaod_table), df)

        # Lets see if the parent can be processed. If not, then we just continue
        # to return false.
        can_process, modified_df = self._process_by_depth(df.parent)
        if not can_process:
            return (can_process, modified_df)

        # Parent can be processed. Thus we can process everything.
        return (True, df)

    def process(self, df: DataFrame) -> Union[DataFrame, result]:
        'Process as much of the tree as we can process'
        can_process, modified_df = self._process_by_depth(df)

        if not can_process:
            return modified_df
        else:
            r = hep_tables.make_local(df)
            return result(r)
