from hl_tables import make_local_async
from hep_tables import xaod_table
from func_adl import EventDataset
import pytest
from .utils_for_testing import hep_tables_make_local_call  # NOQA


@pytest.fixture
def df_truth_count():
    dataset = EventDataset('localds://mc16_13TeV:bogus')
    df = xaod_table(dataset)
    truth = df.TruthParticles('TruthParticles')
    llp_truth = truth[truth.pdgId == 35]
    return llp_truth.Count()


@pytest.mark.asyncio
async def test_hep_table_truth_count(df_truth_count, hep_tables_make_local_call):  # NOQA
    'This should be run totally by hep tables'
    await make_local_async(df_truth_count)
    hep_tables_make_local_call.assert_called_once_with(df_truth_count)
