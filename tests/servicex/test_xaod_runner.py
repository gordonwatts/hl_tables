from dataframe_expressions import DataFrame
from func_adl import EventDataset
from hep_tables import xaod_table
from hl_tables.servicex.xaod_runner import xaod_runner
from hl_tables.runner import result
import pytest
from ..utils_for_testing import hep_tables_make_local_call

@pytest.fixture
def good_xaod():
    e = EventDataset('localds://dude.dataset')
    return xaod_table(e)


def test_with_nothing(hep_tables_make_local_call):
    # Hand it a xAOD that isn't based on anything.
    df = DataFrame()
    x = xaod_runner()
    r = x.process(df)
    assert r is df
    hep_tables_make_local_call.assert_not_called()


def test_with_result(good_xaod, hep_tables_make_local_call):
    # Hand it something it can do, and then see if it can make it work!
    x = xaod_runner()
    df1 = good_xaod.x
    r = x.process(df1)

    assert r is not None
    assert isinstance(r, result)
    assert r.result is not None
    hep_tables_make_local_call.assert_called_once_with(df1)
