import ast
import asyncio
import copy
from typing import Dict, Iterable, Union, Any

from dataframe_expressions import Column, ast_DataFrame
from dataframe_expressions.DataFrame import DataFrame
from dataframe_expressions.asts import ast_Column
import hep_tables
from hep_tables.hep_table import xaod_table

from ..ast_utils import AsyncNodeTransformer
from ..runner import ast_awkward, result, runner


def _has_df_ref(a: ast.AST):
    class find_it(ast.NodeVisitor):
        def __init__(self):
            self.seen = False

        def visit_ast_DataFrame(self, node: ast_DataFrame):
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
        self._df_asts: Dict[int, Union[ast_DataFrame, ast_Column]] = {}
        self._good = True

    def _mark(self, a: ast.AST, is_good: bool):
        self._marks[id(a)] = is_good

    def lookup_mark(self, a: ast.AST) -> bool:
        i = id(a)
        assert i in self._marks, 'internal programming error'
        return self._marks[i]

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

        if df.child_expr is None:
            self._good = isinstance(df, xaod_table)
            return

        self.visit(df.child_expr)

        if df.filter is not None:
            self.visit(self._col_ast_for(df.filter))

    def visit_ast_Column(self, node: ast_Column):
        self.visit(node.column.child_expr)

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        self._update_good(not _check_for_df_ref([node.left, node.right]))

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        self._update_good(not _check_for_df_ref([node.left] + node.comparators))

    def visit_Call(self, node: ast.Call) -> Any:
        'Certain calls are bad'
        self.generic_visit(node)

        is_histogram = (isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'histogram')
        self._update_good(not is_histogram)


class _transform(AsyncNodeTransformer):
    def __init__(self, m: _mark):
        AsyncNodeTransformer.__init__(self)
        self._marker = m
        self._cached_results: Dict[int, ast_awkward] = {}
        self._results_in_progress: Dict[int, asyncio.Event] = {}
        self._cached_lock = asyncio.Lock()

    async def visit_ast_DataFrame(self, node: ast_DataFrame) -> ast.AST:
        try:
            if self._marker.lookup_mark(node):
                async with self._cached_lock:
                    t_id = id(node.dataframe)
                    if t_id in self._cached_results:
                        return self._cached_results[t_id]

                    if t_id in self._results_in_progress:
                        lock = self._results_in_progress[t_id]
                        do_calc = False
                    else:
                        lock = asyncio.Event()
                        self._results_in_progress[t_id] = lock
                        do_calc = True
                if do_calc:
                    r = ast_awkward(await hep_tables.make_local_async(node.dataframe))
                    async with self._cached_lock:
                        self._cached_results[t_id] = r
                        del self._results_in_progress[t_id]
                    lock.set()
                    return r
                else:
                    await lock.wait()
                    return self._cached_results[t_id]

            else:
                # Since it isn't good, we need to traverse the tree of all the dependents
                # to see if we can run anything there. We'll have to alter the "parent"
                # we are passing down, of course, as we've moved a level down in the tree.
                df = node.dataframe
                results = []
                if df.child_expr is not None:
                    results.append(self.visit(df.child_expr))

                if df.filter is not None:
                    results.append(self.visit(self._marker._col_ast_for(df.filter)))

                await asyncio.gather(*results)

                return node
        except Exception as e:
            from dataframe_expressions import dumps
            bad_df = '\n'.join(dumps(node.dataframe))
            raise Exception(f'Internal Error: Failed to render DataFrame: {bad_df}') from e

    async def visit_ast_Column(self, node: ast_Column) -> ast.AST:
        # Since this is never marked "good", we need to only explore the child expressions
        # to see if there is some rendering hidden there.
        # Note we replace the parent as we go down one here with
        # None - there is no parent in ast_DataFrame.
        await self.visit(node.column.child_expr)
        return node


def _as_ast(df: Union[DataFrame, Column]) -> Union[ast_DataFrame, ast_Column]:
    if isinstance(df, DataFrame):
        return ast_DataFrame(df)
    else:
        return ast_Column(df)


async def _process(df: Union[DataFrame, Column]) -> Union[DataFrame, Column, result]:
    '''
    Process as much of the dataframe/column as we can in a two step process.

    1. Mark all the nodes in the tree as executable or not
    2. On the transition from "bad" to "good", execute the nodes.
    '''
    # Mark everything in the tree as either being "good" or bad.
    marker = _mark()
    deep_df = copy.deepcopy(df)
    top_level_ast = _as_ast(deep_df)
    marker.visit(top_level_ast)

    # Run the transformation to see what we can actually convert.
    t = _transform(marker)
    r = await t.visit(top_level_ast)

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
    async def process(self, df: DataFrame) -> Union[DataFrame, Column, result]:
        'Process as much of the tree as we can process'
        return await _process(df)
