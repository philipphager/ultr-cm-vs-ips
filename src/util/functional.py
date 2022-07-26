from functools import reduce
from typing import List, Callable


def apply(fn: List[Callable], value):
    return reduce(lambda v, fn: fn(v), fn, value)
