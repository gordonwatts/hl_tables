import ast
from typing import cast

from dataframe_expressions import DataFrame
from dataframe_expressions.asts import ast_DataFrame

from hl_tables.atlas import a_3v


def test_3v_create():
    df = DataFrame()
    v = a_3v(df)
    assert v is not None


def test_3v_subtract():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 - v2
    assert r is not None
    assert isinstance(r, DataFrame)


def test_3v_add():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 + v2
    assert r is not None
    assert isinstance(r, DataFrame)


def test_3v_x():
    df = DataFrame()
    v1 = a_3v(df)
    df1 = v1.x
    assert df1 is not None
    assert df1.child_expr is not None
    assert ast.dump(df1.child_expr) == "Attribute(value=ast_DataFrame(), " \
                                       "attr='x', ctx=Load())"


def test_3v_y():
    df = DataFrame()
    v1 = a_3v(df)
    df1 = v1.y
    assert df1 is not None
    assert df1.child_expr is not None
    assert ast.dump(df1.child_expr) == "Attribute(value=ast_DataFrame(), " \
                                       "attr='y', ctx=Load())"


def test_3v_z():
    df = DataFrame()
    v1 = a_3v(df)
    df1 = v1.z
    assert df1 is not None
    assert df1.child_expr is not None
    assert ast.dump(df1.child_expr) == "Attribute(value=ast_DataFrame(), " \
                                       "attr='z', ctx=Load())"


def test_3v_add_x():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 + v2
    df1 = r.x
    assert df1 is not None
    assert df1.child_expr is not None
    assert ast.dump(df1.child_expr) == "BinOp(left=ast_DataFrame(), " \
                                       "op=Add(), right=ast_DataFrame())"
    assert isinstance(df1.child_expr, ast.BinOp)
    assert isinstance(df1.child_expr.left, ast_DataFrame)
    df1_parent = cast(ast_DataFrame, df1.child_expr.left).dataframe
    assert df1_parent.child_expr is not None
    assert ast.dump(df1_parent.child_expr) == "Attribute(value=ast_DataFrame(), " \
                                              "attr='x', ctx=Load())"


def test_3v_xy():
    df = DataFrame()
    v1 = a_3v(df)
    df1 = v1.xy
    assert df1.child_expr is not None
    assert ast.dump(df1.child_expr) == "Call(func=Attribute(value=ast_DataFrame()," \
                                       " attr='sqrt', ctx=Load()), args=[], keywords=[])"
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1_parent = cast(ast_DataFrame, df1.child_expr.func.value).dataframe
    assert df1_parent.child_expr is not None
    assert ast.dump(df1_parent.child_expr) == "BinOp(left=ast_DataFrame(), " \
                                              "op=Add(), right=ast_DataFrame())"
    assert isinstance(df1_parent.child_expr, ast.BinOp)
    df1_parent_parent = cast(ast_DataFrame, df1_parent.child_expr.left).dataframe
    assert df1_parent_parent.child_expr is not None
    assert ast.dump(df1_parent_parent.child_expr) == "BinOp(left=ast_DataFrame(), " \
                                                     "op=Mult(), right=ast_DataFrame())"
    assert isinstance(df1_parent_parent.child_expr, ast.BinOp)
    df1_parent_parent_parent = cast(ast_DataFrame, df1_parent_parent.child_expr.left).dataframe
    assert df1_parent_parent_parent.child_expr is not None
    assert ast.dump(df1_parent_parent_parent.child_expr) == \
        "Attribute(value=ast_DataFrame(), attr='x', ctx=Load())"


def test_3v_add_xy():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 + v2
    df1 = r.xy
    assert df1.child_expr is not None
    assert ast.dump(df1.child_expr) == "Call(func=Attribute(value=ast_DataFrame()," \
                                       " attr='sqrt', ctx=Load()), args=[], keywords=[])"
    assert isinstance(df1.child_expr, ast.Call)
    assert isinstance(df1.child_expr.func, ast.Attribute)
    df1_parent = cast(ast_DataFrame, df1.child_expr.func.value).dataframe
    assert df1_parent.child_expr is not None
    assert ast.dump(df1_parent.child_expr) == "BinOp(left=ast_DataFrame(), " \
                                              "op=Add(), right=ast_DataFrame())"
    assert isinstance(df1_parent.child_expr, ast.BinOp)
    df1_parent_parent = cast(ast_DataFrame, df1_parent.child_expr.left).dataframe
    assert df1_parent_parent.child_expr is not None
    assert ast.dump(df1_parent_parent.child_expr) == "BinOp(left=ast_DataFrame(), " \
                                                     "op=Mult(), right=ast_DataFrame())"
    assert isinstance(df1_parent_parent.child_expr, ast.BinOp)
    df1_parent_parent_parent = cast(ast_DataFrame, df1_parent_parent.child_expr.left).dataframe

    assert df1_parent_parent_parent.child_expr is not None
    assert ast.dump(df1_parent_parent_parent.child_expr) == \
        "BinOp(left=ast_DataFrame(), op=Add(), right=ast_DataFrame())"
    assert isinstance(df1_parent_parent_parent.child_expr, ast.BinOp)
    assert isinstance(df1_parent_parent_parent.child_expr.left, ast_DataFrame)
    df1_parent_parent_parent_parent = \
        cast(ast_DataFrame, df1_parent_parent_parent.child_expr.left).dataframe

    assert df1_parent_parent_parent_parent.child_expr is not None
    assert ast.dump(df1_parent_parent_parent_parent.child_expr) == \
        "Attribute(value=ast_DataFrame(), attr='x', ctx=Load())"
