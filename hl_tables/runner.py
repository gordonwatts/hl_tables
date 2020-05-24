from abc import abstractmethod
from typing import Any, Union
import ast

from dataframe_expressions import DataFrame


class result:
    '''
    A result returned by a processor.
    TODO: Do we need this, or is the awkward df all we need (or similar)?
    '''
    def __init__(self, r: object):
        self._result = r

    @property
    def result(self):
        return self._result


class awkward_DataFrame(DataFrame):
    # TODO: COuld we delete this?
    def __init__(self, awk: Any):
        self.awk = awk


class ast_awkward(ast.AST):
    _fields = ('awkward', )

    def __init__(self, a: Any):
        self.awkward = a


class runner:
    '''
    Base class for any runner that can help with the DAG built by `dataframe_expressions`
    '''

    @abstractmethod
    def process(self, df: DataFrame) -> Union[DataFrame, result]: pass
