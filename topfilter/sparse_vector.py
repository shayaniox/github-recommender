import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

class SparseVector:
    def __init__(self, N: int):
        self.N = N
        self.st = {}  # Using dict instead of ST class for simplicity

    def reset(self):
        self.st = {}

    def put(self, i: int, value: float):
        if i < 0 or i >= self.N:
            raise ValueError(f"Illegal index {i} for vector size {self.N}")
        if value == 0.0:
            self.st.pop(i, None)
        else:
            self.st[i] = value

    def get(self, i: int) -> float:
        if i < 0 or i >= self.N:
            raise ValueError(f"Illegal index {i} for vector size {self.N}")
        return self.st.get(i, 0.0)

    def nnz(self) -> int:
        return len(self.st)

    def size(self) -> int:
        return self.N

    def dot(self, b: 'SparseVector') -> float:
        if self.N != b.N:
            raise ValueError("Vector lengths disagree")
        
        sum_val = 0.0
        if len(self.st) <= len(b.st):
            for i in self.st:
                if i in b.st:
                    sum_val += self.get(i) * b.get(i)
        else:
            for i in b.st:
                if i in self.st:
                    sum_val += self.get(i) * b.get(i)
        return sum_val

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def scale(self, alpha: float) -> 'SparseVector':
        c = SparseVector(self.N)
        for i in self.st:
            c.put(i, alpha * self.get(i))
        return c

    def cosine_similarity(self, b: 'SparseVector') -> float:
        if self.N != b.N:
            raise ValueError("Vector lengths disagree")
        
        sum_val = 0.0
        val1 = sum(v * v for v in self.st.values())
        val2 = sum(v * v for v in b.st.values())
        
        if len(self.st) <= len(b.st):
            for i in self.st:
                if i in b.st:
                    sum_val += self.get(i) * b.get(i)
        else:
            for i in b.st:
                if i in self.st:
                    sum_val += self.get(i) * b.get(i)
        
        if sum_val == 0:
            return 0.0
        return sum_val / math.sqrt(val1 * val2)

    def plus(self, b: 'SparseVector') -> 'SparseVector':
        if self.N != b.N:
            raise ValueError("Vector lengths disagree")
        
        c = SparseVector(self.N)
        for i in self.st:
            c.put(i, self.get(i))
        for i in b.st:
            c.put(i, b.get(i) + c.get(i))
        return c

    def __str__(self) -> str:
        return " ".join(f"({i}, {val})" for i, val in self.st.items())

    def test(self):
        a = SparseVector(10)
        b = SparseVector(10)
        a.put(3, 0.50)
        a.put(9, 0.75)
        a.put(6, 0.11)
        a.put(6, 0.00)
        b.put(3, 0.60)
        b.put(4, 0.90)
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"a dot b = {a.dot(b)}")
        print(f"a + b = {a.plus(b)}")

