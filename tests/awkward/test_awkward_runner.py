# type: ignore
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
def awk_arr_uniform():
    return awkward.fromiter([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])


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


@pytest.mark.asyncio
async def test_no_awkward_root():
    df = DataFrame()

    ar = awkward_runner()
    r = await ar.process(df)
    assert isinstance(r, DataFrame)


@pytest.mark.asyncio
async def test_addition(awk_arr_pair):
    df = DataFrame()
    df1 = df + df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_addition_one(awk_arr):
    df = DataFrame()
    df1 = df + 1000.0

    n1 = awk_arr

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_subtraction(awk_arr_pair):
    df = DataFrame()
    df1 = df - df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_multiplication(awk_arr_pair):
    df = DataFrame()
    df1 = df * df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_division(awk_arr_pair):
    df = DataFrame()
    df1 = df / df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_numpy_sqrt(awk_arr_pair):
    import numpy as np
    df = DataFrame()
    df1 = np.sqrt(df + df)

    n1, n2 = awk_arr_pair

    assert df1.parent is not None
    assert isinstance(df1.parent.child_expr, ast.BinOp)
    df1.parent.child_expr.left = ast_awkward(n1)
    df1.parent.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_addition_twice(awk_arr_3):
    df = DataFrame()
    df1 = df + df + df

    n1, n2, n3 = awk_arr_3

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.right = ast_awkward(n1)

    assert isinstance(df1.parent.child_expr, ast.BinOp)
    df1.parent.child_expr.left = ast_awkward(n1)
    df1.parent.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_histogram(awk_arr_pair):
    df = DataFrame()
    df1 = df + df

    n1, n2 = awk_arr_pair

    assert isinstance(df1.child_expr, ast.BinOp)
    df1.child_expr.left = ast_awkward(n1)
    df1.child_expr.right = ast_awkward(n2)

    xr = awkward_runner()
    r = await xr.process(histogram(df1, bins=50, range=(0, 20)))

    assert isinstance(r, result)


@pytest.mark.asyncio
async def test_count(awk_arr):
    df = DataFrame()
    df1 = df.Count()

    n1 = awk_arr

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 5
    assert a[0] == 3
    assert a[1] == 0
    assert a[2] == 2


@pytest.mark.asyncio
async def test_count_axis_0(awk_arr_uniform):
    df = DataFrame()
    df1 = df.Count(axis=0)

    n1 = awk_arr_uniform

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert isinstance(a, int)
    assert a == 3


@pytest.mark.asyncio
async def test_count_axis_last(awk_arr_uniform):
    df = DataFrame()
    df1 = df.Count(axis=-1)

    n1 = awk_arr_uniform

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 3
    assert a[0] == 2
    assert a[1] == 2
    assert a[2] == 2


@pytest.mark.asyncio
async def test_count_toplevel(awk_arr_onelevel):
    df = DataFrame()
    df1 = df.Count()

    n1 = awk_arr_onelevel

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert a == 4


@pytest.mark.asyncio
async def test_count_toplevel_axis_4(awk_arr_onelevel):
    df = DataFrame()
    df1 = df.Count(axis=4)

    n1 = awk_arr_onelevel

    assert df1.parent is not None
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1.child_expr.func.value = ast_awkward(n1)

    xr = awkward_runner()
    with pytest.raises(Exception) as e:
        await xr.process(df1)

    assert "axis" in str(e.value)


@pytest.mark.asyncio
async def test_compare_single_number_eq(awk_arr_onelevel):
    df = DataFrame()
    df1 = df == 2

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr_onelevel)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 4
    assert sum(a) == 1
    assert a[1]
    assert not a[0]


@pytest.mark.asyncio
async def test_compare_single_number_ne(awk_arr_onelevel):
    df = DataFrame()
    df1 = df != 2

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr_onelevel)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 4
    assert sum(a) == 3
    assert not a[1]
    assert a[0]


@pytest.mark.asyncio
async def test_compare_single_number_gt(awk_arr_onelevel):
    df = DataFrame()
    df1 = df > 2

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr_onelevel)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 4
    assert sum(a) == 2
    assert a[3]
    assert not a[0]


@pytest.mark.asyncio
async def test_compare_single_number_lt(awk_arr_onelevel):
    df = DataFrame()
    df1 = df < 2

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr_onelevel)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 4
    assert sum(a) == 1
    assert a[0]
    assert not a[3]


@pytest.mark.asyncio
async def test_compare_single_number_ge(awk_arr_onelevel):
    df = DataFrame()
    df1 = df >= 2

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr_onelevel)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 4
    assert sum(a) == 3
    assert a[1]
    assert not a[0]


@pytest.mark.asyncio
async def test_compare_single_number_le(awk_arr_onelevel):
    df = DataFrame()
    df1 = df <= 2

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr_onelevel)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert len(a) == 4
    assert sum(a) == 2
    assert a[1]
    assert not a[3]


@pytest.mark.asyncio
async def test_compare_single_number_eq_two(awk_arr):
    df = DataFrame()
    df1 = df == 1.1

    assert isinstance(df1.child_expr, ast.Compare)
    df1.child_expr.left = ast_awkward(awk_arr)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None
    assert sum(a.flatten()) == 1
    assert a[0][0]


@pytest.mark.asyncio
async def test_filter_simple(awk_arr):
    df = DataFrame()
    df1 = df[df == 3.3]

    assert isinstance(df1.filter.child_expr, ast.Compare)
    df1.filter.child_expr.left = ast_awkward(awk_arr)

    df1.child_expr = ast_awkward(awk_arr)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None

    assert len(a) == 5
    assert len(a.flatten()) == 1


@pytest.mark.asyncio
async def test_mapseq_index_filter(awk_arr_uniform):
    df = DataFrame()
    df1 = df.mapseq(lambda s: s[0] == 3.3)

    df1.child_expr.func.value = ast_awkward(awk_arr_uniform)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None

    assert len(a) == 3
    assert len(a.flatten()) == 3
    assert sum(a.flatten()) == 1
    assert a[1]


@pytest.mark.asyncio
async def test_mapseq_eval_parent(awk_arr_uniform):
    df = DataFrame()
    df1 = df+1.0
    df2 = df1.mapseq(lambda s: s[1] == 3.2)

    df1.child_expr.left = ast_awkward(awk_arr_uniform)

    xr = awkward_runner()
    r = await xr.process(df2)

    assert isinstance(r, result)
    a = r.result
    assert a is not None

    assert len(a) == 3
    assert len(a.flatten()) == 3
    assert sum(a.flatten()) == 1
    assert a[0]


@pytest.mark.asyncio
async def test_mapseq_index_filter_with_and(awk_arr_uniform):
    df = DataFrame()
    df1 = df.mapseq(lambda s: (s[0] == 3.3) & (s[1] == 4.4))

    df1.child_expr.func.value = ast_awkward(awk_arr_uniform)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None

    assert len(a) == 3
    assert len(a.flatten()) == 3
    assert sum(a.flatten()) == 1
    assert a[1]


@pytest.mark.asyncio
async def test_mapseq_index_filter_with_or(awk_arr_uniform):
    df = DataFrame()
    df1 = df.mapseq(lambda s: (s[0] == 3.3) | (s[1] == 6.6))

    df1.child_expr.func.value = ast_awkward(awk_arr_uniform)

    xr = awkward_runner()
    r = await xr.process(df1)

    assert isinstance(r, result)
    a = r.result
    assert a is not None

    assert len(a) == 3
    assert len(a.flatten()) == 3
    assert sum(a.flatten()) == 2
    assert a[1]
    assert a[2]
