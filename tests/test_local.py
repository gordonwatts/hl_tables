import ast
from typing import Any

from func_adl import EventDataset
from hep_tables import xaod_table
import pytest

from hl_tables import make_local_async

from .utils_for_testing import hep_tables_make_local_call  # NOQA


@pytest.fixture
def df_truth_count():
    class my_events(EventDataset):
        async def execute_result_async(self, a: ast.AST) -> Any:
            raise NotImplementedError()
    df = xaod_table(my_events())
    truth = df.TruthParticles('TruthParticles')
    llp_truth = truth[truth.pdgId == 35]
    return llp_truth.Count()


@pytest.mark.asyncio
async def test_hep_table_truth_count(df_truth_count, hep_tables_make_local_call):  # NOQA
    'This should be run totally by hep tables'
    await make_local_async(df_truth_count)
    hep_tables_make_local_call.assert_called_once_with(df_truth_count)
