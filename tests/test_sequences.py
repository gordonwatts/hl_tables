import pytest
import asyncio

from hl_tables import count
from dataframe_expressions import DataFrame


@pytest.fixture
def call_make_local_count(mocker):
    'Return a dummy histogram from a make local call'

    f = asyncio.Future()
    f.set_result(20)

    return mocker.patch('hl_tables.local.make_local_async', return_value=f)


@pytest.mark.asyncio
async def test_histogram(call_make_local_count):
    df = DataFrame()
    r = await count(df)
    assert r == 20

    call_make_local_count.assert_called_once()
