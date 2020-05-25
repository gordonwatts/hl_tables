import pytest
import asyncio


@pytest.fixture
def hep_tables_make_local_call(mocker):
    r = asyncio.Future()
    r.set_result(mocker.MagicMock())
    return mocker.patch('hep_tables.make_local_async', return_value=r)


@pytest.fixture
def hep_tables_make_local_call_pause(mocker):
    'Make sure to pause a little bit - enough for things to pile up'
    async def wait_for_it():
        await asyncio.sleep(0.1)
        return mocker.MagicMock()

    return mocker.patch('hep_tables.make_local_async', side_effect=lambda m: wait_for_it())
