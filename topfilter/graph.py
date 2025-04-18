import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

class Graph:
    def __init__(self, filename: Optional[str] = None, dictionary: Optional[Dict[int, str]] = None):
        self.OutLinks: Dict[int, Set[int]] = defaultdict(set)
        self.nodeCount = 0
        self.dictionary: Dict[str, int] = {}

        if filename is not None:
            if dictionary is not None:
                self._init_with_dictionary(filename, dictionary)
            else:
                self._init_from_file(filename)

    def _init_from_file(self, filename: str):
        nodes = set()
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split(",")
                    for pairs in vals:
                        pair = pairs.split("#")
                        start_node = int(pair[0].strip())
                        end_node = int(pair[1].strip())
                        nodes.add(start_node)
                        nodes.add(end_node)
                        self.OutLinks[start_node].add(end_node)
            self.nodeCount = len(nodes)
        except IOError as e:
            logging.error(f"Error initializing graph from {filename}: {e}")

    def _init_with_dictionary(self, filename: str, dictionary: Dict[int, str]):
        nodes = set()
        key_set = set(dictionary.keys())
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    line = line.strip()
                    pair = line.split("#")
                    start_node = int(pair[0].strip())
                    end_node = int(pair[1].strip())

                    if start_node in key_set and end_node in key_set:
                        nodes.add(start_node)
                        nodes.add(end_node)
                        self.OutLinks[start_node].add(end_node)
            self.nodeCount = len(nodes)
        except IOError as e:
            logging.error(f"Error initializing graph with dictionary from {filename}: {e}")

    def get_out_links(self) -> Dict[int, Set[int]]:
        return self.OutLinks

    def set_out_links(self, out_links: Dict[int, Set[int]]):
        self.OutLinks = out_links

    def set_num_nodes(self, n: int):
        self.nodeCount = n

    def get_num_nodes(self) -> int:
        return self.nodeCount

    def get_dictionary(self) -> Dict[str, int]:
        return self.dictionary

    def combine(self, graph: 'Graph', dictionary: Dict[int, str]):
        tmp_out_links = graph.get_out_links()
        main_outlinks = set()

        for start_node, outlinks in tmp_out_links.items():
            artifact = dictionary.get(start_node, "")
            id_start_node = self._extract_key(artifact)

            for end_node in outlinks:
                artifact = dictionary.get(end_node, "")
                id_end_node = self._extract_key(artifact)

                if id_start_node in self.OutLinks:
                    main_outlinks = self.OutLinks[id_start_node]
                else:
                    main_outlinks = set()

                main_outlinks.add(id_end_node)
                self.OutLinks[id_start_node] = main_outlinks

        nodes = set()
        for start_node, outlinks in self.OutLinks.items():
            nodes.add(start_node)
            nodes.update(outlinks)

        self.nodeCount = len(nodes)

    def _extract_key(self, s: str) -> int:
        if s in self.dictionary:
            return self.dictionary[s]
        else:
            c = len(self.dictionary)
            self.dictionary[s] = c
            return c

    def out_links(self, id: int) -> Set[int]:
        return self.OutLinks.get(id, set())

    def num_nodes(self) -> int:
        return self.nodeCount

