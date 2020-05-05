from hl_tables.atlas import a_3v
from dataframe_expressions import DataFrame


def test_3v_create():
    df = DataFrame()
    v = a_3v(df)
    assert v is not None
