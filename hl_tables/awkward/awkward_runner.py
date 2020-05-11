import ast
import logging
from typing import Union

from dataframe_expressions import (
    Column,
    DataFrame,
    ast_Callable,
    ast_DataFrame,
    ast_Filter,
    render,
    render_callable,
    render_context,
)
import numpy as np

from ..runner import ast_awkward, result, runner


def _replace_dataframe(source: ast.AST, df: DataFrame, replacement: ast.AST):
    class rep(ast.NodeTransformer):
        def visit_ast_DataFrame(self, node: ast_DataFrame):
            if node.dataframe is df:
                return replacement
            return node

    return rep().visit(source)


class inline_executor(ast.NodeTransformer):
    'Inline execute'
    def __init__(self, context: render_context):
        ast.NodeTransformer.__init__(self)
        self._context = context

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Num, ast.Eq)):
            return node

        a = ast.NodeTransformer.visit(self, node)
        if not isinstance(a, ast_awkward):
            logging \
                .getLogger(__name__) \
                .warning(f'Unable to awkward array calculate {ast.dump(node)}')
        return a

    def call_mapseq(self, node: ast.Call, value: ast_awkward) -> ast.AST:
        assert len(node.args) == 1, 'mapseq takes only one argument'
        c_func = node.args[0]
        if not isinstance(c_func, ast_Callable):
            print('func is not callable')
            return node

        # Generate the expression using a dataframe ast to get generic behavior
        # Replace it with the awkward array in the end
        a = DataFrame()
        df_expr, new_context = render_callable(c_func, self._context, a)
        expr = _replace_dataframe(df_expr, a, value)

        # Just run it through this processor to continue the evaluation.
        old_context, self._context = self._context, new_context
        try:
            print("going to visit expr")
            e = self.visit(expr)
            if not isinstance(e, ast_awkward):
                print(f'mapseq eval failure: {ast.dump(expr)}')
                print(f'mapseq eval got back: {ast.dump(e)}')
            return e
        finally:
            self._context = old_context

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if node is None:
            return node

        if not isinstance(node.func, ast.Attribute):
            return node

        r = self.visit(node.func.value)

        if not isinstance(r, ast_awkward):
            return node

        o = r.awkward

        if node.func.attr == 'sqrt':
            return ast_awkward(np.sqrt(o))
        elif node.func.attr == 'Count':
            if isinstance(o, np.ndarray):
                return ast_awkward(len(o))
            else:
                return ast_awkward(o.count())
        elif node.func.attr == 'histogram':
            if hasattr(o, 'flatten'):
                o = o.flatten()
            kwargs = {n.arg: ast.literal_eval(n.value) for n in node.keywords}
            hist_info = np.histogram(o, **kwargs)
            return ast_awkward(hist_info)
        elif node.func.attr == 'mapseq':
            return self.call_mapseq(node, r)
        else:
            return node

    def visit_BinOp(self, r: ast.BinOp) -> ast.AST:
        # If this isn't in the right format, there isn't much we can do.
        node = self.generic_visit(r)
        o1 = node.left.awkward if isinstance(node.left, ast_awkward) else ast.literal_eval(node.left)
        o2 = node.left.awkward if isinstance(node.right, ast_awkward) else ast.literal_eval(node.right)

        if isinstance(node.op, ast.Add):
            return ast_awkward(o1 + o2)
        elif isinstance(node.op, ast.Sub):
            return ast_awkward(o1 - o2)
        elif isinstance(node.op, ast.Mult):
            return ast_awkward(o1 * o2)
        elif isinstance(node.op, ast.Div):
            return ast_awkward(o1 / o2)
        else:
            return r

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        assert len(node.comparators) == 1, 'Internal error - cannot do compare of more complex than 2 operands'
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])

        if not isinstance(left, ast_awkward):
            return node
        o = left.awkward

        r_value = ast.literal_eval(right)

        if isinstance(node.ops[0], ast.Eq):
            return ast_awkward(o == r_value)
        elif isinstance(node.ops[0], ast.NotEq):
            return ast_awkward(o != r_value)
        elif isinstance(node.ops[0], ast.Gt):
            return ast_awkward(o > r_value)
        elif isinstance(node.ops[0], ast.Lt):
            return ast_awkward(o < r_value)
        elif isinstance(node.ops[0], ast.GtE):
            return ast_awkward(o >= r_value)
        elif isinstance(node.ops[0], ast.LtE):
            return ast_awkward(o <= r_value)

        return node

    def visit_ast_Filter(self, node: ast_Filter) -> ast.AST:
        expr = self.visit(node.expr)
        filter = self.visit(node.filter)

        if not isinstance(expr, ast_awkward):
            return node
        if not isinstance(filter, ast_awkward):
            return node

        return ast_awkward(expr.awkward[filter.awkward])

    def visit_Subscript(self, node: ast.Subscript):
        if not isinstance(node.value, ast_awkward):
            return node
        a = node.value.awkward
        if not isinstance(node.slice, ast.Index):
            return node
        index = node.slice.value

        return ast_awkward(a[..., index])

    def visit_BoolOp(self, node: ast.BoolOp):
        assert len(node.values) == 2, 'Can only do simple binary operator compares'
        left = self.visit(node.values[0])
        right = self.visit(node.values[1])
        assert isinstance(left, ast_awkward)
        assert isinstance(right, ast_awkward)

        if isinstance(node.op, ast.And):
            return ast_awkward(left.awkward & right.awkward)
        elif isinstance(node.op, ast.Or):
            return ast_awkward(left.awkward | right.awkward)

        return node


class awkward_runner(runner):
    '''
    We can do a xaod on servicex
    '''
    def process(self, df: DataFrame) -> Union[DataFrame, Column, result]:
        'Process as much of the tree as we can process'

        # Render it into an ast that we can now process!
        r, context = render(df)

        calc = inline_executor(context).visit(r)

        if isinstance(calc, ast_awkward):
            return result(calc.awkward)

        return df
