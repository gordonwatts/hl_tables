from hl_tables.atlas import a_3v
from dataframe_expressions import DataFrame


def test_3v_create():
    df = DataFrame()
    v = a_3v(df)
    assert v is not None


def test_3v_subtract():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 - v2
    assert r is not None
    assert isinstance(r, DataFrame)
