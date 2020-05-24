from hl_tables.ast_utils import AsyncNodeTransformer
import ast
import pytest
import asyncio
from typing import Any


@pytest.mark.asyncio
async def test_atrans_simple():
    a = ast.parse('x+1')
    r = await AsyncNodeTransformer().visit(a)
    assert ast.dump(a) == ast.dump(r)


@pytest.mark.asyncio
async def test_atrans_dual():
    a = ast.parse('x + 10 if x < 10 else x - 10')
    r = await AsyncNodeTransformer().visit(a)
    assert ast.dump(a) == ast.dump(r)


@pytest.mark.asyncio
async def test_atrans_replace():
    class repl(AsyncNodeTransformer):
        async def visit_Num(self, node: ast.Num, context: Any):
            await asyncio.sleep(0.01)
            assert node.n == 10
            return ast.Num(n=20)

    a = ast.parse('x + 10 if x < 10 else x - 10')
    r = await repl().visit(a)
    a2 = ast.parse('x + 20 if x < 20 else x - 20')
    assert ast.dump(a2) == ast.dump(r)


@pytest.mark.asyncio
async def test_atrans_context():
    class repl(AsyncNodeTransformer):
        async def visit_Num(self, node: ast.Num, context: Any):
            assert context == 10
            return node

    a = ast.parse('x + 10 if x < 10 else x - 10')
    await repl().visit(a, 10)
