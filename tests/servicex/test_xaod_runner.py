import ast
from typing import cast

from dataframe_expressions import Column, DataFrame
from func_adl import EventDataset
from hep_tables import xaod_table
import pytest

from hl_tables.runner import ast_awkward, result
from hl_tables.servicex.xaod_runner import xaod_runner

from ..utils_for_testing import hep_tables_make_local_call  # NOQA

# For dev work do: pip install git+https://github.com/numpy/numpy-stubs


@pytest.fixture
def good_xaod():
    e = EventDataset('localds://dude.dataset')
    return xaod_table(e)


def test_with_nothing(hep_tables_make_local_call):  # NOQA
    # Hand it a xAOD that isn't based on anything.
    df = DataFrame()
    x = xaod_runner()
    r = x.process(df)
    assert r is df
    hep_tables_make_local_call.assert_not_called()


def test_with_nothing_deeper(hep_tables_make_local_call):  # NOQA
    # Hand it a xAOD that isn't based on anything.
    df = DataFrame()
    df1 = df[df.x > 10].y
    x = xaod_runner()
    r = x.process(df1)
    assert r is df1
    hep_tables_make_local_call.assert_not_called()


def test_with_result(good_xaod, hep_tables_make_local_call):  # NOQA
    # Hand it something it can do, and then see if it can make it work!
    x = xaod_runner()
    df1 = good_xaod.x
    r = x.process(df1)

    assert r is not None
    assert isinstance(r, result)
    assert r.result is not None
    hep_tables_make_local_call.assert_called_once_with(df1)


def test_with_filter_result(good_xaod, hep_tables_make_local_call):  # NOQA
    # Hand it something it can do, and then see if it can make it work!
    x = xaod_runner()
    df1 = good_xaod[good_xaod.x > 10].x
    r = x.process(df1)

    assert r is not None
    assert isinstance(r, result)
    assert r.result is not None
    hep_tables_make_local_call.assert_called_once_with(df1)


def test_split_call_binary_const(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x * 10
    r = x.process(df1)
    assert r is not None
    assert isinstance(r, result)
    assert hep_tables_make_local_call.call_count == 1


def test_split_call_binary(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x + good_xaod.y
    r = x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    a = r.child_expr
    assert isinstance(a, ast.BinOp)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.right, ast_awkward)


def test_split_call_compare(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    df1 = good_xaod.x > good_xaod.y
    r = x.process(df1)
    assert r is not None
    assert isinstance(r, Column)
    assert hep_tables_make_local_call.call_count == 2
    a = r.child_expr
    assert isinstance(a, ast.Compare)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.comparators[0], ast_awkward)


def test_split_call_good_filter(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    ok_events = good_xaod[good_xaod.x > 10]
    df1 = ok_events.x + ok_events.y
    r = x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    a = r.child_expr
    assert isinstance(a, ast.BinOp)
    assert isinstance(a.left, ast_awkward)
    assert isinstance(a.right, ast_awkward)


def test_split_in_filter(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    ok_events = good_xaod[good_xaod.x > good_xaod.y]
    df1 = ok_events.z
    r = x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 3


def test_split_in_sqrt(good_xaod, hep_tables_make_local_call):  # NOQA
    x = xaod_runner()
    import numpy as np
    df1 = cast(DataFrame, np.sqrt(good_xaod.x + good_xaod.y))
    r = x.process(df1)
    assert r is not None
    assert isinstance(r, DataFrame)
    assert hep_tables_make_local_call.call_count == 2
    assert isinstance(r.child_expr, ast.Call)
