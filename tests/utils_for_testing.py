import pytest


@pytest.fixture
def hep_tables_make_local_call(mocker):
    return mocker.patch('hep_tables.make_local')
