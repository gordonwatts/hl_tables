from abc import abstractmethod
from typing import Any, Union, Optional
import ast

from dataframe_expressions import DataFrame


class result:
    '''
    A result returned by a processor.
    '''
    def __init__(self, r: object):
        self._result = r

    @property
    def result(self):
        return self._result


class awkward_DataFrame(DataFrame):
    def __init__(self, awk: Any):
        self.awk = awk


class ast_awkward(ast.AST):
    _fields = ('awkward', )

    def __init__(self, a: Optional[Any] = None):
        self.awkward = a


class runner:
    '''
    Base class for any runner that can help with the DAG built by `dataframe_expressions`
    '''

    @abstractmethod
    async def process(self, df: DataFrame) -> Union[DataFrame, result]:
        pass
