# Copyright (C) 2023 Alexandre Jesus <https://adbjesus.com>, Carlos M. Fonseca <cmfonsec@dei.uc.pt>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import  Optional,Tuple, Iterable
from math import ceil, log2
import random

def non_repeating_lcg(n: int, seed: Optional[int] = None) -> Iterable[int]:
    if seed is not None:
        random.seed(seed)
    "Pseudorandom sampling without replacement in O(1) space"
    if n > 0:
        a = 5 # always 5
        m = 1 << ceil(log2(n))
        if m > 1:
            c = random.randrange(1, m, 2)
            x = random.randrange(m)
            for _ in range(m):
                if x < n: yield x
                x = (a * x + c) % m
        else:
            yield 0

def sample(n: int, seed: Optional[int] = None) -> Iterable[int]:
    for v in non_repeating_lcg(n, seed):
        yield v

def sample2(n: int, m: int, seed: Optional[int] = None) -> Iterable[Tuple[int, int]]:
    for idx, v in enumerate(non_repeating_lcg(n * m, seed)):
        i = v // m
        j = v % m + 1
        yield i, j