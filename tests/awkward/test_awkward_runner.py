import ast
from hl_tables.runner import ast_awkward, result
from hep_tables import histogram

import awkward
from dataframe_expressions.DataFrame import DataFrame
import pytest

from hl_tables.awkward.awkward_runner import awkward_runner


@pytest.fixture()
def awk_arr():
    n1 = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    return n1


@pytest.fixture()
def awk_arr_onelevel():
    n1 = awkward.fromiter([1, 2, 3, 4])
    return n1

@pytest.fixture()
def awk_arr_pair():
    n1 = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    n2 = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    return (n1, n2)


@pytest.fixture()
def awk_arr_3():
    n1 = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    n2 = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    n3 = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    return (n1, n2, n3)


def test_no_awkward_root():
    df = DataFrame()

    ar = awkward_runner()
    r = ar.process(df)
    assert isinstance(r, DataFrame)


def test_addition(awk_arr_pair):
    df = DataFrame()
    df1 = df + df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_addition_one(awk_arr):
    df = DataFrame()
    df1 = df + 1000.0

    n1 = awk_arr

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_subtraction(awk_arr_pair):
    df = DataFrame()
    df1 = df - df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_multiplication(awk_arr_pair):
    df = DataFrame()
    df1 = df * df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_division(awk_arr_pair):
    df = DataFrame()
    df1 = df / df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_numpy_sqrt(awk_arr_pair):
    import numpy as np
    df = DataFrame()
    df1 = np.sqrt(df + df)

    n1, n2 = awk_arr_pair

    assert df1.parent is not None
    assert isinstance(df1.parent.child_expr, ast.BinOp)
    df1.parent.child_expr.left = ast_awkward(n1)
    df1.parent.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_addition_twice(awk_arr_3):
    df = DataFrame()
    df1 = df + df + df

    n1, n2, n3 = awk_arr_3

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.right = ast_awkward(n1)

    assert isinstance(df1.parent.child_expr, ast.BinOp)
    df1.parent.child_expr.left = ast_awkward(n1)
    df1.parent.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)


def test_histogram(awk_arr_pair):
    df = DataFrame()
    df1 = df + df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = xr.process(histogram(df1, bins=50, range=(0, 20)))

    assert isinstance(r, result)


def test_count(awk_arr):
    df = DataFrame()
    df1 = df.Count()

    n1 = awk_arr

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 5
    assert a[0] == 3
    assert a[1] == 0
    assert a[2] == 2


def test_count_toplevel(awk_arr_onelevel):
    df = DataFrame()
    df1 = df.Count()

    n1 = awk_arr_onelevel

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    r = xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert a == 4
