from typing import Union, Tuple, Optional
import ast

from dataframe_expressions import DataFrame, ast_DataFrame, Column

from ..runner import result, runner, ast_awkward
import hep_tables

# TODO: this should not be in the hl_tables package, but it its own or in
#       in the `hep_tables` package.


def has_df_ref(a: ast.AST):
    class find_it(ast.NodeVisitor):
        def __init__(self):
            self.seen = False

        def visit_ast_DataFrame(self, node: ast_DataFrame):
            self.seen = True

        def visit_Name(self, node: ast.Name):
            if node.id == 'p':
                self.seen = True

    f = find_it()
    f.visit(a)
    return f.seen


def _can_process(a: Optional[ast.AST]) -> bool:
    'Is this an AST we can deal with?'

    # Does the ast contain a binary operation, where each side refers to an iterator?
    # Servicex can process this, but `hep_tables` can't.

    class detect_binary(ast.NodeVisitor):
        def __init__(self):
            self.bad_binary = False

        def check_args(self, asts):
            count = sum([1 for a in asts if has_df_ref(a)])
            if count > 1:
                self.bad_binary = True

        def visit_BinOp(self, node: ast.BinOp):
            self.generic_visit(node)
            self.check_args([node.left, node.right])

        def visit_Compare(self, node: ast.Compare):
            self.generic_visit(node)
            self.check_args([node.left] + node.comparators)

    if a is None:
        return True

    d = detect_binary()
    d.visit(a)
    return not d.bad_binary


def _process_ast(a: Optional[ast.AST], parent: DataFrame) -> Optional[ast.AST]:
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

    if a is None:
        return None

    return process(parent).visit(a)


class xaod_runner(runner):
    '''
    We can do a xaod on servicex
    '''
    def _process_by_depth(self, df: Union[DataFrame, Column]) \
            -> Tuple[bool, Union[DataFrame, Column, result]]:
        'Process by depth'

        # Lets do the really easy thing first - what happens when we are at the bottom.
        if isinstance(df, DataFrame):
            if df.parent is None:
                assert df.child_expr is None, 'Internal Programming Error'
                return (isinstance(df, hep_tables.xaod_table), df)

            # Lets see if the parent can be processed. If not, then we just continue
            # to return false.
            can_process, modified_df = self._process_by_depth(df.parent)
            if not can_process:
                df.parent = modified_df
                return (can_process, df)

            # Parent can be processed. This is great. Lets check if the child expression
            # can be processed.
            if not _can_process(df.child_expr) or not _can_process(df.filter):
                # Process everything in the expression we can
                new_child = _process_ast(df.child_expr, df.parent)
                new_filter = _process_ast(df.filter, df.parent)
                df.child_expr = new_child
                df.filter = new_filter
                return (False, df)

            return (True, df)
        elif isinstance(df, Column):
            if not _can_process(df.child_expr):
                df.child_expr = _process_ast(df.child_expr, None)
                return (False, df)
            return (True, df)
        else:
            raise Exception('Internal programing error')

    def process(self, df: DataFrame) -> Union[DataFrame, result]:
        'Process as much of the tree as we can process'
        can_process, modified_df = self._process_by_depth(df)

        if not can_process:
            return modified_df
        else:
            r = hep_tables.make_local(df)
            return result(r)
