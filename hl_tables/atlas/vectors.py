from __future__ import annotations
import ast

from dataframe_expressions import DataFrame


class _atlas_3v_ast(ast.AST):
    '3Vector from an atlas dataframe'
    def __init__(self, seq: ast.AST):
        '''
        Create an ast that represents a 3-Vector
        '''
        self.base_seq = seq
        self._fields = ('base_seq', )


def a_3v(df: DataFrame):
    '''
    Return a 3-vector object from the current data frame. The x,y, and z components
    are assumed to be in the (df.x, df.y, df.z) components.
    '''
    return DataFrame(df, expr=_atlas_3v_ast(ast.Name(id='p')))
