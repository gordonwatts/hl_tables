import ast
import logging
from typing import Union

from dataframe_expressions import Column, DataFrame, render, ast_Filter
import numpy as np

from ..runner import ast_awkward, result, runner


class inline_executor(ast.NodeTransformer):
    'Inline execute'

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Num, ast.Eq)):
            return node

        a = ast.NodeTransformer.visit(self, node)
        if not isinstance(a, ast_awkward):
            logging \
                .getLogger(__name__) \
                .warning(f'Unable to awkward array calculate {ast.dump(node)}')
        return a

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
        else:
            return node

    def visit_ast_Filter(self, node: ast_Filter) -> ast.AST:
        expr = self.visit(node.expr)
        filter = self.visit(node.filter)

        if not isinstance(expr, ast_awkward):
            return node
        if not isinstance(filter, ast_awkward):
            return node

        return ast_awkward(expr.awkward[filter.awkward])


class awkward_runner(runner):
    '''
    We can do a xaod on servicex
    '''
    def process(self, df: DataFrame) -> Union[DataFrame, Column, result]:
        'Process as much of the tree as we can process'

        # Render it into an ast that we can now process!
        r, _ = render(df)

        calc = inline_executor().visit(r)

        if isinstance(calc, ast_awkward):
            return result(calc.awkward)

        return df
