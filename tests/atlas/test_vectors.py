import ast

from dataframe_expressions import DataFrame
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
    assert ast.dump(df1.child_expr) == "Attribute(value=Name(id='p', ctx=Load()), " \
                                       "attr='x', ctx=Load())"


def test_3v_add_x():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 + v2
    df1 = r.x
    assert df1 is not None
    assert ast.dump(df1.child_expr) == "BinOp(left=Name(id='p', ctx=Load()), " \
                                       "op=Add(), right=ast_DataFrame())"
    assert ast.dump(df1.parent.child_expr) == "Attribute(value=Name(id='p', ctx=Load()), " \
                                              "attr='x', ctx=Load())"


def test_3v_xy():
    df = DataFrame()
    v1 = a_3v(df)
    df1 = v1.xy
    assert ast.dump(df1.child_expr) == "Call(func=Attribute(value=Name(id='p', ctx=Load())," \
                                       " attr='sqrt', ctx=Load()), args=[], keywords=[])"
    assert ast.dump(df1.parent.child_expr) == "BinOp(left=Name(id='p', ctx=Load()), " \
                                              "op=Add(), right=ast_DataFrame())"
    assert ast.dump(df1.parent.parent.child_expr) == "BinOp(left=Name(id='p', ctx=Load()), " \
                                                     "op=Mult(), right=ast_DataFrame())"
    assert ast.dump(df1.parent.parent.parent.child_expr) == \
        "Attribute(value=Name(id='p', ctx=Load()), attr='x', ctx=Load())"


def test_3v_add_xy():
    df = DataFrame()
    v1 = a_3v(df)
    v2 = a_3v(df)
    r = v1 + v2
    df1 = r.xy
    assert ast.dump(df1.child_expr) == "Call(func=Attribute(value=Name(id='p', ctx=Load())," \
                                       " attr='sqrt', ctx=Load()), args=[], keywords=[])"
    assert ast.dump(df1.parent.child_expr) == "BinOp(left=Name(id='p', ctx=Load()), " \
                                              "op=Add(), right=ast_DataFrame())"
    assert ast.dump(df1.parent.parent.child_expr) == "BinOp(left=Name(id='p', ctx=Load()), " \
                                                     "op=Mult(), right=ast_DataFrame())"
    assert ast.dump(df1.parent.parent.parent.child_expr) == \
        "BinOp(left=Name(id='p', ctx=Load()), op=Add(), right=ast_DataFrame())"
    assert ast.dump(df1.parent.parent.parent.parent.child_expr) == \
        "Attribute(value=Name(id='p', ctx=Load()), attr='x', ctx=Load())"
