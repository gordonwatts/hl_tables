import ast
from typing import Any

from func_adl.EventDataset import EventDataset
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


@pytest.mark.asyncio
async def test_missed_table_monad_reference():
    '''
    This distilled bug query was found in the wild - it seem
    the system can't track the lambda captured reference to
    an xaod_table.
    '''
    class my_events(EventDataset):
        async def execute_result_async(self, a: ast.AST) -> Any:
            import awkward
            return {b'col1': awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])}

    df = xaod_table(my_events())

    # Logging means that we will get more info about what
    # is going on - helps with debugging.
    import logging
    ch = logging.StreamHandler()
    logging.getLogger('dataframe_expressions').setLevel(logging.DEBUG)
    logging.getLogger('dataframe_expressions').addHandler(ch)

    def associate_particles(source, pick_from, name: str):
        source['all_LLP'] = lambda source_p: pick_from[lambda ps: 2.0 < ps.eta()]
        source['is_LLP'] = lambda e: e.all_LLP.Count() > 0
        return source[source.is_LLP]

    mc_part = df.TruthParticles('TruthParticles')
    df['loose_jets_LLP'] = associate_particles(df.jets, mc_part, 'LLP')

    await make_local_async(df.loose_jets_LLP.pt)
