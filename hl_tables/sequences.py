from dataframe_expressions import DataFrame


async def count(df: DataFrame) -> int:
    '''
    Given a dataframe, it will return an int at the outter most level. And run everything too,
    and return it.
    '''
    from hl_tables.local import make_local_async
    return await make_local_async(df.Count(axis=0))
