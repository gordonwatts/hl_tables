from typing import Union, Tuple
import ast

from dataframe_expressions import DataFrame, ast_DataFrame

from ..runner import result, runner, ast_awkward
import hep_tables

# TODO: this should not be in the hl_tables package, but it its own or in
#       in the `hep_tables` package.


def _can_process(a: ast.AST) -> bool:
    'Is this an AST we can deal with?'

    # Does the ast contain a binary operation, where each side refers to an iterator?
    # Servicex can process this, but `hep_tables` can't.

    class detect_binary(ast.NodeVisitor):
        def __init__(self):
            self.bad_binary = False

        def visit_BinOp(self, node: ast.BinOp):
            if _can_process(node.left) and _can_process(node.right):
                self.bad_binary = True
            else:
                self.generic_visit(node)

    d = detect_binary()
    d.visit(a)
    return not d.bad_binary


def _process_ast(a: ast.AST, parent: DataFrame) -> ast.AST:
    'Process everything we can in the ast'

    class process(ast.NodeTransformer):
        def __init__(self, p: DataFrame):
            self._parent = p

        def make_local_df(self, df: DataFrame) -> ast_awkward:
            r = hep_tables.make_local(df)
            return ast_awkward(r)

        def visit_Name(self, a: ast.Name):
            if a.id == 'p':
                return self.make_local_df(self._parent)
            return a

        def visit_ast_DataFrame(self, a: ast_DataFrame):
            return self.make_local_df(a.dataframe)

    return process(parent).visit(a)


class xaod_runner(runner):
    '''
    We can do a xaod on servicex
    '''
    def _process_by_depth(self, df: DataFrame) -> Tuple[bool, Union[DataFrame, result]]:
        'Process by depth'

        # Lets do the really easy thing first - what happens when we are at the bottom.
        if df.parent is None:
            assert df.child_expr is None, 'Internal Programming Error'
            return (isinstance(df, hep_tables.xaod_table), df)

        # Lets see if the parent can be processed. If not, then we just continue
        # to return false.
        can_process, modified_df = self._process_by_depth(df.parent)
        if not can_process:
            return (can_process, modified_df)

        # Parent can be processed. This is great. Lets check if the child expression
        # can be processed.
        if not _can_process(df.child_expr):
            # Process everything in the expression we can
            new_expr = _process_ast(df.child_expr, df.pare)
            df.child_expr = new_expr
            return (False, df)

        return (True, df)

    def process(self, df: DataFrame) -> Union[DataFrame, result]:
        'Process as much of the tree as we can process'
        can_process, modified_df = self._process_by_depth(df)

        if not can_process:
            return modified_df
        else:
            r = hep_tables.make_local(df)
            return result(r)
