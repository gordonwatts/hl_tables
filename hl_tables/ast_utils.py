from ast import AST, iter_fields
import asyncio
from typing import Any, Optional


class AsyncNodeVisitor:
    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.
    This class is meant to be subclassed, with the subclass adding visitor
    methods.
    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `visit` method.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.
    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing.  For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    async def visit(self, node: AST):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return await visitor(node)

    async def generic_visit(self, node: AST, context: Optional[Any] = None):
        """Called if no explicit visitor function exists for a node."""
        for _, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        await self.visit(item)
            elif isinstance(value, AST):
                await self.visit(value)


class AsyncNodeTransformer(AsyncNodeVisitor):
    """
    A :class:`NodeVisitor` subclass that walks the abstract syntax tree and
    allows modification of nodes.
    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace or remove the old node.  If the return value of
    the visitor method is ``None``, the node will be removed from its location,
    otherwise it is replaced with the return value.  The return value may be the
    original node in which case no replacement takes place.
    Here is an example transformer that rewrites all occurrences of name lookups
    (``foo``) to ``data['foo']``::
       class RewriteName(NodeTransformer):
           def visit_Name(self, node):
               return Subscript(
                   value=Name(id='data', ctx=Load()),
                   slice=Constant(value=node.id),
                   ctx=node.ctx
               )
    Keep in mind that if the node you're operating on has child nodes you must
    either transform the child nodes yourself or call the :meth:`generic_visit`
    method for the node first.
    For nodes that were part of a collection of statements (that applies to all
    statement nodes), the visitor may also return a list of nodes rather than
    just a single node.
    Usually you use the transformer like this::
       node = YourTransformer().visit(node)
    """

    async def generic_visit(self, node: AST):
        async def eval_single_field(field, old_value):
            async def eval_list_item(value, new_values):
                if isinstance(value, AST):
                    value = await self.visit(value)
                    if value is None:
                        return
                    elif not isinstance(value, AST):
                        new_values.extend(value)
                        return
                new_values.append(value)

            if isinstance(old_value, list):
                new_values = []
                results = [eval_list_item(value, new_values) for value in old_value]
                await asyncio.gather(*results)
                old_value[:] = new_values

            elif isinstance(old_value, AST):
                new_node = await self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)

        results = [eval_single_field(field, old_value) for field, old_value in iter_fields(node)]
        await asyncio.gather(*results)

        return node
