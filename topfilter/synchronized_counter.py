import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

class SynchronizedCounter:
    def __init__(self, value: int = -1):
        self.c = value
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.c += 1

    def decrement(self):
        with self._lock:
            self.c -= 1

    def value(self) -> int:
        with self._lock:
            self.increment()
            return self.c
