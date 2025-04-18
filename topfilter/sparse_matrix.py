import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

class SparseMatrix:
    def __init__(self, N: int):
        self.N = N
        self.cols = [SparseVector(N) for _ in range(N)]

    def put(self, i: int, j: int, value: float):
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            raise ValueError("Illegal index")
        self.cols[i].put(j, value)

    def get(self, i: int, j: int) -> float:
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            raise ValueError("Illegal index")
        return self.cols[i].get(j)

    def nnz(self) -> int:
        return sum(col.nnz() for col in self.cols)

    def times(self, x: SparseVector) -> SparseVector:
        if self.N != x.size():
            raise ValueError("Dimensions disagree")
        b = SparseVector(self.N)
        for i in range(self.N):
            b.put(i, self.cols[i].dot(x))
        return b

    def plus(self, B: 'SparseMatrix') -> 'SparseMatrix':
        if self.N != B.N:
            raise ValueError("Dimensions disagree")
        C = SparseMatrix(self.N)
        for i in range(self.N):
            C.cols[i] = self.cols[i].plus(B.cols[i])
        return C

    def __str__(self) -> str:
        s = f"N = {self.N}, nonzeros = {self.nnz()}\n"
        for i in range(self.N):
            s += f"{i}: {self.cols[i]}\n"
        return s

    def test(self):
        A = SparseMatrix(5)
        x = SparseVector(5)
        A.put(0, 0, 1.0)
        A.put(1, 1, 1.0)
        A.put(2, 2, 1.0)
        A.put(3, 3, 1.0)
        A.put(4, 4, 1.0)
        A.put(2, 4, 0.3)
        x.put(0, 0.75)
        x.put(2, 0.11)
        print(f"x: {A.get(3, 4)}")
