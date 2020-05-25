import ast
from typing import cast

from dataframe_expressions import Column, DataFrame
from func_adl import EventDataset
from hep_tables import xaod_table
import pytest

from hl_tables.runner import ast_awkward, result
from hl_tables.servicex.xaod_runner import xaod_runner

from ..utils_for_testing import hep_tables_make_local_call, hep_tables_make_local_call_pause  # NOQA

# For dev work do: pip install git+https://github.com/numpy/numpy-stubs


@pytest.fixture
def good_xaod():
    e = EventDataset('localds://dude.dataset')
    return xaod_table(e)


@pytest.mark.asyncio
async def test_with_nothing(hep_tables_make_local_call):  # NOQA
    # Hand it a xAOD that isn't based on anything.
    df = DataFrame()
    x = xaod_runner()
    r = await x.process(df)
    assert isinstance(r, DataFrame)
    hep_tables_make_local_call.assert_not_called()


@pytest.mark.asyncio
async def test_with_nothing_deeper(hep_tables_make_local_call):  # NOQA
    # Hand it a xAOD that isn't based on anything.
    df = DataFrame()
    df1 = df[df.x > 10].y
    x = xaod_runner()
    r = await x.process(df1)
    assert isinstance(r, DataFrame)
    hep_tables_make_local_call.assert_not_called()


@pytest.mark.asyncio
async def test_with_result(good_xaod, hep_tables_make_local_call):  # NOQA
    # Hand it something it can do, and then see if it can make it work!
    x = xaod_runner()
    df1 = good_xaod.x
    r = await x.process(df1)

    assert r is not None
    assert isinstance(r, result)
    assert r.result is not None
    hep_tables_make_local_call.assert_called_once_with(df1)


@pytest.mark.asyncio
async def test_with_filter_result(good_xaod, hep_tables_make_local_call):  # NOQA
    # Hand it something it can do, and then see if it can make it work!
    x = xaod_runner()
    df1 = good_xaod[good_xaod.x > 10].x
    r = await x.process(df1)

    assert r is not None
    assert isinstance(r, result)
    assert r.result is not None
    hep_tables_make_local_call.assert_called_once_with(df1)


@pytest.mark.asyncio
async def test_split_call_binary_const(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x * 10
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, result)
    assert hep_tables_make_local_call.call_count == 1


@pytest.mark.asyncio
async def test_split_call_binary(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x + good_xaod.y
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    a = r.child_expr
    assert isinstance(a, ast.BinOp)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.right, ast_awkward)


@pytest.mark.asyncio
async def test_split_call_compare(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x > good_xaod.y
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, Column)
    assert hep_tables_make_local_call.call_count == 2
    a = r.child_expr
    assert isinstance(a, ast.Compare)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.comparators[0], ast_awkward)


@pytest.mark.asyncio
async def test_split_double(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x + good_xaod.y
    r = await x.process(df1[df1 < 20])
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2


@pytest.mark.asyncio
async def test_split_double_count(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x + good_xaod.y
    r = await x.process(df1[df1 < 20].Count())
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2


@pytest.mark.asyncio
async def test_split_call_good_filter(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    ok_events = good_xaod[good_xaod.x > 10]
    df1 = ok_events.x + ok_events.y
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    a = r.child_expr
    assert isinstance(a, ast.BinOp)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.right, ast_awkward)


@pytest.mark.asyncio
async def test_split_in_filter(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    ok_events = good_xaod[good_xaod.x > good_xaod.y]
    df1 = ok_events.z
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 3


@pytest.mark.asyncio
async def test_split_in_sqrt(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    import numpy as np
    df1 = cast(DataFrame, np.sqrt(good_xaod.x + good_xaod.y))   # type: ignore
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    assert isinstance(r.child_expr, ast.Call)


@pytest.mark.asyncio
async def test_split_in_sqrt_with_divide(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    import numpy as np
    df1 = cast(DataFrame, np.sqrt(good_xaod.x + good_xaod.y)/1000.0)   # type: ignore
    r = await x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    assert isinstance(r.child_expr, ast.BinOp)


@pytest.mark.asyncio
async def test_run_twice(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    ok_events = good_xaod[good_xaod.x > 10]
    df1 = ok_events.x + ok_events.y
    _ = await x.process(df1)

    r2 = await x.process(df1)

    assert r2 is not None
    assert isinstance(r2, DataFrame)
    assert hep_tables_make_local_call.call_count == 4
    a = r2.child_expr
    assert isinstance(a, ast.BinOp)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.right, ast_awkward)


@pytest.mark.asyncio
async def test_request_twice(good_xaod, hep_tables_make_local_call_pause):  # NOQA
    x = xaod_runner()
    ok_events = good_xaod[good_xaod.x > 10]
    df1 = ok_events.x + ok_events.y + ok_events.x
    _ = await x.process(df1)

    assert hep_tables_make_local_call_pause.call_count == 2


@pytest.mark.asyncio
async def test_run_mapseq(good_xaod, hep_tables_make_local_call):  # NOQA
    from hl_tables.atlas import a_3v
    truth = good_xaod.TruthParticles('TruthParticles')
    llp_truth = truth[truth.pdgId == 35]
    llp_good_truth = llp_truth[llp_truth.hasProdVtx & llp_truth.hasDecayVtx]
    l_prod = a_3v(llp_good_truth.prodVtx)
    l_decay = a_3v(llp_good_truth.decayVtx)
    lxy = (l_decay-l_prod).xy
    lxy_2 = lxy[lxy.Count() == 2]
    has_1muon = lxy_2[lxy_2.mapseq(lambda s: s[0] > 1 | s[1] < 2)].Count()

    x = xaod_runner()
    r = await x.process(has_1muon)

    assert isinstance(r, DataFrame)

    all_args = hep_tables_make_local_call.call_args_list
    for a in all_args:
        d = a[0][0]
        from dataframe_expressions import render
        r, _ = render(d)
        print(ast.dump(r))
    assert len(all_args) == 4
