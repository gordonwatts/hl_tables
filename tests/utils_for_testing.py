import pytest
import asyncio


@pytest.fixture
def hep_tables_make_local_call(mocker):
    r = asyncio.Future()
    r.set_result(mocker.MagicMock())
    return mocker.patch('hep_tables.make_local', return_value=r)
