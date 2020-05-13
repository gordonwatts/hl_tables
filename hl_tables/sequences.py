from dataframe_expressions import DataFrame


def count(df: DataFrame) -> int:
    '''
    Given a dataframe, it will return an int at the outter most level. And run everything too, and return it.
    '''
    from hl_tables import make_local
    return make_local(df.Count(axis=0))
