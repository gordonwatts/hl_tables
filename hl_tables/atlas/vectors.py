
from dataframe_expressions import DataFrame


class _atlas_3v:
    '3Vector from an atlas dataframe'
    def __init__(self, df: DataFrame):
        self._df = df


def a_3v(df: DataFrame):
    '''
    Return a 3-vector object from the current data frame. The x,y, and z components
    are assumed to be in the (df.x, df.y, df.z) components.
    '''
    return _atlas_3v(df)
