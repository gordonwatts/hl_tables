from abc import abstractmethod
from dataframe_expressions import DataFrame
from typing import Union


class result:
    '''
    A result returned by a processor.
    '''
    def __init__(self, r: object):
        self._result = r

    @property
    def result(self):
        return self._result


class runner:
    '''
    Base class for any runner that can help with the DAG built by `dataframe_expressions`
    '''

    @abstractmethod
    def process(self, df: DataFrame) -> Union[DataFrame, result]: pass
