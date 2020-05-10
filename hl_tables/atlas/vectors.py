from __future__ import annotations
from typing import Callable, Any, Optional, cast

from dataframe_expressions import DataFrame, exclusive_class


# TODO: does this really belong in this package in the end?

class op_base:
    def render(self, f: Callable[[_atlas_3v], Any]) -> Any:
        assert False, 'not implemented'


class op_bin(op_base):
    def __init__(self, op: str, a: op_base, b: op_base):
        self._a = a
        self._b = b
        self._op = op

    def render(self, f: Callable[[_atlas_3v], Any]) -> Any:
        if self._op == '+':
            return self._a.render(f) + self._b.render(f)
        elif self._op == '-':
            return self._a.render(f) - self._b.render(f)
        assert False, f'Unknown vector operation {self._op}'


class op_vec(op_base):
    def __init__(self, df: DataFrame):
        self._df = df

    def render(self, f: Callable[[DataFrame], Any]) -> Any:
        return f(self._df)


@exclusive_class
class _atlas_3v(DataFrame):
    def __init__(self, df: DataFrame, compound: Optional[op_base] = None) -> None:
        DataFrame.__init__(self, df)
        self._ref: op_base = compound if compound is not None else op_vec(df)

    @property
    def xy(self) -> DataFrame:
        from numpy import sqrt
        bx = self._ref.render(lambda v: v.x)
        by = self._ref.render(lambda v: v.y)
        return cast(DataFrame, sqrt(bx*bx + by*by))

    @property
    def x(self) -> DataFrame:
        return self._ref.render(lambda v: v.x)

    @property
    def y(self) -> DataFrame:
        return self._ref.render(lambda v: v.y)

    @property
    def z(self) -> DataFrame:
        return self._ref.render(lambda v: v.z)

    def __add__(self, other: _atlas_3v) -> _atlas_3v:
        'Do the addition'
        assert isinstance(other, _atlas_3v)
        assert self.parent is not None
        return _atlas_3v(self.parent, op_bin('+', self._ref, other._ref))

    def __sub__(self, other: _atlas_3v) -> _atlas_3v:
        'Do the addition'
        assert isinstance(other, _atlas_3v)
        assert self.parent is not None
        return _atlas_3v(self.parent, op_bin('-', self._ref, other._ref))


def a_3v(df: DataFrame):
    '''
    Return a 3-vector object from the current data frame. The x,y, and z components
    are assumed to be in the (df.x, df.y, df.z) components.
    '''
    return _atlas_3v(df)
