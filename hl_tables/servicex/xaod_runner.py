from typing import Union, Dict, Iterable, Optional
import ast

from dataframe_expressions import DataFrame, ast_DataFrame, Column
from dataframe_expressions.asts import ast_Column
from hep_tables.hep_table import xaod_table

from ..runner import result, runner, ast_awkward
import hep_tables

# TODO: this should not be in the hl_tables package, but it its own or in
#       in the `hep_tables` package.


def _has_df_ref(a: ast.AST):
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


def _check_for_df_ref(asts: Iterable[ast.AST]) -> bool:
    'Anyone have more than one reference to some sort of DF?'
    return sum([1 for a in asts if _has_df_ref(a)]) > 1


class _mark(ast.NodeVisitor):
    def __init__(self):
        ast.NodeVisitor.__init__(self)
        self._marks: Dict[int, bool] = {}
        self._parent: Optional[DataFrame] = None
        self._df_asts: Dict[int, Union[ast_DataFrame, ast_Column]] = {}
        self._good = True

    def _mark(self, a: ast.AST, is_good: bool):
        self._marks[id(a)] = is_good

    def lookup_mark(self, a: ast.AST) -> bool:
        i = id(a)
        assert i in self._marks, 'internal programming error'
        return self._marks[i]

    def _df_ast_for(self, df: DataFrame) -> ast.AST:
        'Cache the ast dataframe object so we can track its good and bad'
        h = id(df)
        if h not in self._df_asts:
            self._df_asts[h] = ast_DataFrame(df)
        return self._df_asts[h]

    def _col_ast_for(self, c: Column) -> ast.AST:
        h = hash(c)
        if c not in self._df_asts:
            self._df_asts[h] = ast_Column(c)
        return self._df_asts[h]

    def _update_good(self, is_good: bool):
        'Only update good if we are still positive'
        if self._good:
            self._good = is_good

    def visit(self, node: ast.AST):
        'Track if this node is good or not'
        old_good = self._good

        self._good = True
        ast.NodeVisitor.visit(self, node)
        self._mark(node, self._good)

        # If that was good, then we want the same status we came in with
        # (be it good or bad)
        self._update_good(old_good)

    def visit_ast_DataFrame(self, node: ast_DataFrame):
        # Look for a top level dataframe we can't deal with.
        df = node.dataframe
        if df.parent is None:
            self._good = isinstance(df, xaod_table)
            return

        # Ok - we have to go down one level here, sadly.
        old_parent = self._parent
        self._parent = df.parent

        if df.child_expr is not None:
            self.visit(df.child_expr)
        else:
            self.visit(self._df_ast_for(df.parent))

        if df.filter is not None:
            self.visit(self._col_ast_for(df.filter))

        self._parent = old_parent

    def visit_ast_Column(self, node: ast_Column):
        old_parent = self._parent
        self._parent = None

        self.visit(node.column.child_expr)

        self._parent = old_parent

    def visit_Name(self, node: ast.Name):
        if node.id == 'p':
            assert self._parent is not None
            self.visit(self._df_ast_for(self._parent))

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        self._update_good(not _check_for_df_ref([node.left, node.right]))

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        self._update_good(not _check_for_df_ref([node.left] + node.comparators))


class _transform(ast.NodeTransformer):
    def __init__(self, m: _mark):
        ast.NodeTransformer.__init__(self)
        self._marker = m
        self._parent = None

    def visit_ast_DataFrame(self, node: ast_DataFrame) -> ast.AST:
        if self._marker.lookup_mark(node):
            r = hep_tables.make_local(node.dataframe)
            return ast_awkward(r)
        else:
            df = node.dataframe

            # Ok - we have to go down one level here, sadly.
            old_parent = self._parent
            self._parent = df.parent

            if df.child_expr is not None:
                self.visit(df.child_expr)
            elif df.parent is not None:
                self.visit(self._marker._df_ast_for(df.parent))

            if df.filter is not None:
                self.visit(self._marker._col_ast_for(df.filter))

            self._parent = old_parent
            return node

    def visit_ast_Column(self, node: ast_Column) -> ast.AST:
        old_parent = self._parent
        self._parent = None

        self.visit(node.column.child_expr)

        self._parent = old_parent
        return node

    def visit_Name(self, node: ast.Name):
        if node.id == 'p':
            assert self._parent is not None
            return self.visit(self._marker._df_ast_for(self._parent))
        return node


def _as_ast(df: Union[DataFrame, Column]) -> Union[ast_DataFrame, ast_Column]:
    if isinstance(df, DataFrame):
        return ast_DataFrame(df)
    else:
        return ast_Column(df)


def _process(df: Union[DataFrame, Column]) -> Union[DataFrame, Column, result]:
    '''
    Process as much of the dataframe/column as we can in a two step process.

    1. Mark all the nodes in the tree as executable or not
    2. On the transition from "bad" to "good", execute the nodes.
    '''
    # Mark everything in the tree as either being "good" or bad.
    marker = _mark()
    top_level_ast = _as_ast(df)
    marker.visit(top_level_ast)

    # Run the transformation to see what we can actually convert.
    t = _transform(marker)
    r = t.visit(top_level_ast)

    if isinstance(r, ast_Column):
        return r.column
    elif isinstance(r, ast_DataFrame):
        return r.dataframe
    elif isinstance(r, ast_awkward):
        return result(r.awkward)
    else:
        assert False, 'should never return something arbitrary!'


class xaod_runner(runner):
    '''
    We can do a xaod on servicex
    '''
    def process(self, df: DataFrame) -> Union[DataFrame, Column, result]:
        'Process as much of the tree as we can process'
        return _process(df)
        # can_process, modified_df = _process_by_depth(df)

        # if not can_process:
        #     return modified_df
        # else:
        #     r = hep_tables.make_local(df)
        #     return result(r)
