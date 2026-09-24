"""Repo-to-topic bipartite graph, ported from the Java CrossRec tool.

RUNNING EXAMPLE. `graph_freeCodeCamp___freeCodeCamp` holds a star from the repo node to each
of its topic nodes, and `dicth_freeCodeCamp___freeCodeCamp` names them:

    graph file            dicth file
    1#2                   1  freeCodeCamp___freeCodeCamp
    1#3                   2  #DEP#learn
    1#4                   3  #DEP#nodejs
    ...                   4  #DEP#react   ...

Per-file IDs are local: every repo's own node is ID 1. `combine` is what maps them into one
global ID space, keyed by the artifact STRING, so '#DEP#nodejs' gets the same global ID no
matter which repo file it arrived from.
"""

import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

# Graph.py
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
        """Load a graph file, keeping only edges whose BOTH endpoints are in `dictionary`.

        That filter is what lets a test repo's star be reused with substituted labels. Its
        graph file was written from 8 real topics:

            file        1#2 1#3 1#4 1#5 1#6 1#7 1#8 1#9
            dictionary  {1: repo, 2..6: the 5 CNBN seeds}     <- only keys 1..6

            kept        1#2 1#3 1#4 1#5 1#6        <- relabelled onto the seeds
            dropped     1#7 1#8 1#9                <- no key, silently discarded

        So the topology is recycled and the labels swapped. If CNBN returned fewer than 5
        topics, or the repo had fewer real topics than seeds, edges vanish here without a
        warning.
        """
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
        """Merge another repo's graph into this one, interning artifact names to global IDs.

            self already holds    bravo___llm-gateway -> {#DEP#python, #DEP#api, #DEP#docker}
            incoming graph        1#2 1#3 with dictionary {1: echo___fastapi-starter,
                                                           2: #DEP#python, 3: #DEP#api}

            _extract_key('#DEP#python') finds the existing global ID and reuses it;
            '#DEP#async', unseen, gets a fresh one.

            result   two repo nodes, sharing the #DEP#python and #DEP#api topic nodes

        After combining ~579 training repos plus one test repo, `OutLinks` has one entry per
        REPO (topic nodes have no outgoing edges), which is why the IDF step can use
        `len(graph_edges)` as the repository count.
        """
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

