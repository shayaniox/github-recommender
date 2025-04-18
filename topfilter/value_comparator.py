import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

class ValueComparator:
    def __init__(self, base):
        self.base = base

    def __call__(self, a, b):
        if self.base[a] > self.base[b]:
            return -1
        elif self.base[a] == self.base[b]:
            return 0
        else:
            return 1
