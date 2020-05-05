from dataframe_expressions import DataFrame
from hl_tables import histogram
import pytest


@pytest.fixture
def call_make_local_hist(mocker):
    'Return a dummy histogram from a make local call'

    def rtn_hist():
        bins = (0.0, 1.0, 2.0)
        data = (20, 30)
        return data, bins

    import hl_tables
    # mocker.patch('hl_tables.local.make_local', side_effect=lambda df: rtn_hist())
    mocker.patch.object(hl_tables.local, 'make_local', side_effect=lambda df: rtn_hist())


def test_histogram(call_make_local_hist):
    # WORKS
    from hl_tables.local import make_local
    dummy = make_local(None)

    # histogram eventually calls make_local - and this does not call my mock!
    df = DataFrame()
    histogram(df, bins=50, range=(0, 20))
