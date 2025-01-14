import os, sys, pickle
prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)
import helper_functions.obo_parser as op
import helper_functions.serialize as serialize

import typing as T
import collections as clt
import itertools as it
import functools as ft
import enum
import bisect

import pandas as pd
import numpy as np


def calculate_depth(ont: op.Ontology, start_term: str):
    d = 0
    term_depth = {start_term: d}
    q: T.Deque = clt.deque()
    q.append(start_term)
    dlst = [d]
    while(len(q) > 0):
      start_term = q.popleft()
      d = dlst.pop(0)
      if len(chiln := ont.ont[start_term]["children"]) > 0:
        chiln = [x for x in chiln if x not in term_depth]
        dlst += [d+1] * len(chiln)
        term_depth.update([(x,d+1) for x in chiln])
        q.extend(chiln)
    return term_depth

def extract_namspace_terms(ont: op.Ontology, data: pd.DataFrame):
  namespace_terms: T.Dict[str, T.Set[str]]
  namespace_terms = {
     "cellular_component": set(),
     "molecular_function": set(),
     "biological_process": set()
  }
  assert "prop_annotations" in data.columns, \
    "a column named prop_annotations not in this dataframe"
  
  NT = T.Tuple[str, str] # Namespace and Term
  anno_namespace_term: pd.Series
  anno_namespace_term = data["prop_annotations"].apply(
       lambda p: [(ont.get_namespace(x), x) 
                  for x in p]) # type: ignore
  nts: T.Iterable[NT] = it.chain.from_iterable(anno_namespace_term)
  for namespace, term in nts:
     assert namespace in namespace_terms, \
         f"{namespace} not in namespace_terms"
     namespace_terms[namespace].add(term)
      
  return namespace_terms

def building(ont: op.Ontology, data: pd.DataFrame):
  new_nt: T.Dict[str, T.List[str]] = {}
  namespace_terms = extract_namspace_terms(ont, data)
  calc_dep = ft.partial(calculate_depth, ont)
  while(namespace_terms):
     namespace, terms = namespace_terms.popitem()
     size = len(terms)
     term_depth = calc_dep(op.NAMESPACE_TERM[namespace])
     get_depth: T.Callable[[str], int]
     get_depth = lambda x: term_depth.get(x,size)
     new_nt[namespace] = sorted(terms, key=get_depth)
  return new_nt

def build_adj_by_depth(ont: op.Ontology, start_term: str):
  """
  Return: adj_r, adj, term_depth
  """
  t: T.Deque = clt.deque()
  t.append(start_term)
  p: T.Dict[str, str] = {}
  p_r: T.Dict[str, T.List[str]] = {}
  d = 0
  term_depth = {start_term: d}
  dlst: T.List[int] = [d]
  while (len(t) > 0):
    start_term = t.popleft()
    d = dlst.pop(0)
    if len(chiln := ont.ont[start_term]["children"]) > 0:
      chiln = [x for x in chiln if x not in p]
      dlst += [d+1] * len(chiln)
      p_r[start_term] = chiln
      t.extend(chiln)
      p.update({x: start_term for x in chiln})
      term_depth.update([(x,d+1) for x in chiln])
  
  return p, p_r, term_depth

class IC(enum.Enum):
  Low = 0
  High = 1

class Position(enum.Enum):
  Shallow = 0
  Medium = 1
  Deep = 2

def topologicalSort(ont: op.Ontology, 
                    namespace: str | None = None,
                    term_lst: T.List[str] | None = None):
  assert namespace is not None or term_lst is not None
  # get size
  if term_lst is None:
    assert namespace is not None
    term_lst = [x for x in ont.get_namespace_terms(namespace)
                             if not ont.ont[x]["is_obsolete"]]

  # Initialize the visited array and stack
  visted = {x: False for x in term_lst}
  stack: T.List[str] = []
  term_set = set(term_lst)
  
  # Perform topological sorting
  for v in term_lst:
    if visted[v]:
      continue
    
    # Create a stack to store the nodes to visit
    nodes_to_visit = [v]
    while nodes_to_visit: # not []
      curr = nodes_to_visit[-1]
      visted[curr] = True
      unvisited_neighbors_exit = False
      
      for neighbor in term_set.intersection(ont.ont[curr]["children"]):
        assert isinstance(neighbor, str)
        if not visted[neighbor]:
          unvisited_neighbors_exit = True
          nodes_to_visit.append(neighbor)
          break
      
      if not unvisited_neighbors_exit:
        stack.append(nodes_to_visit.pop())
  return stack[::-1]

def findLongestPath(ont: op.Ontology, start_term: str,
                    topological_order: T.List[str] | None = None):
  ns_name = ont.get_namespace(start_term)
  if topological_order is None:
    topological_order = topologicalSort(ont, ns_name)
  term_lst: T.List[str] = [x for x in ont.get_namespace_terms(ns_name)
                           if not ont.ont[x]["is_obsolete"]]

  # Initialize distances
  dist = {x: float('-inf') for x in term_lst}
  dist[start_term] = 0

  # Process vertices in topological order
  for u in topological_order:
    if dist[u] != float('-inf'):
      for v in ont.ont[u]["children"]:
        if (d := dist[u] + 1) > dist[v]:
          dist[v] = d

  return dist

def build_ontology_position(ont: op.Ontology, start_term: str,
                            need_longest_path: bool = False,
                            need_maxdep: bool = False):
  ns_name = ont.get_namespace(start_term)
  topological_order = topologicalSort(ont, ns_name)
  longest_path = findLongestPath(ont, start_term,
                                 topological_order)
  term_maxdep: T.Dict[str, float] = {}
  visted = {x: False for x in topological_order}

  for u in topological_order:
    ts = [u]
    branch = []
    while ts:
      start_term = ts.pop(0)
      d = None
      if not visted[start_term]:
        branch.append(start_term)
        visted[start_term] = True
        if (chiln := [x for x in ont.ont[start_term]["children"]]):
          ts = chiln + ts
        else:
          # leaf node
          d = longest_path[start_term]

      else:
        # visited node
        d = term_maxdep[start_term]
      
      if d is not None:
        # update branch
        for x in branch:
          if x not in term_maxdep or \
            term_maxdep[x] < d:
            term_maxdep[x] = d

      # pop node unless to the common parent
      if ts and ts[0] not in ont.ont[branch[-1]]["children"]:
        i = len(branch) - 1
        while i >= 0 and \
          ts[0] not in ont.ont[branch[i]]["children"]:
          i -= 1
        branch = branch[:i+1]
  
  term_position: T.Dict[str, Position] = {}
  poslist = list(Position)
  # partition = lambda x: [x / 3, x * 2 / 3, x]
  partition = np.linspace(1, 3, 3) / 3
  for u in topological_order:
    de_part = partition * term_maxdep[u]
    d = longest_path[u]
    get_idx = ft.partial(bisect.bisect_left, de_part)
    term_position[u] = poslist[get_idx(d)]
  
  match (need_longest_path, need_maxdep):
    case (True, True):
      return term_position, longest_path, term_maxdep
    case (True, False):
      return term_position, longest_path
    case (False, True):
      return term_position, term_maxdep
    case _:
      return term_position  

class OntologyDAG(object):
  def __init__(self, ont: op.Ontology,
               term_ic: T.Dict[str, float],
              #  term_depth: T.Dict[str, int],
               term_lst: T.List[str] | None = None):
    self.term_lst = list(term_ic.keys()) if term_lst is None else term_lst
    self.term_set = set(self.term_lst)
    self.num_terms = len(self.term_lst)
    self.term_idx = {x: i for i, x in enumerate(self.term_lst)}
    self.ont: T.Dict = ont.ont
    self.term_ic = term_ic
    self.toplogical_order = topologicalSort(ont, term_lst=self.term_lst)
    # self.term_depth = term_depth
    # self.depth_lst = [term_depth[x] for x in self.term_lst]
    self.ic_lst = [term_ic[x] for x in self.term_lst]
    self.parents = [max([self.term_idx[x] for x in ts],
                        key=lambda x: self.ic_lst[x])
                    if len(ts := self.term_set.intersection(
                                  self.ont[self.term_lst[i]]["is_a"])) > 0
                    else -1
                    for i in range(self.num_terms)]
    self.children: T.Dict[int, T.List[int]] = {i: [] for i in range(self.num_terms)}
    for i in range(self.num_terms):
        if (x := self.parents[i]) != -1:
          self.children[x].append(i)
  
    self.precompute()

    self.term_depth: T.Dict[int, int] = {}
    self.visited = [False] * self.num_terms
    for u in self.toplogical_order:
      x = self.term_idx[u]
      if self.visited[x]: continue
      self.calculate_depth(x)
    self.depth_lst = [self.term_depth[x] for x in range(self.num_terms)]

  def calculate_depth(self, start_term: int):
    # Calculate the single-source shortest path from start_term by cycling through the
    # parents of each node until the root is reached.
    # This is a width-first search.
    d = 0
    self.term_depth[start_term] = d
    q: T.Deque = clt.deque()
    q.append(start_term)
    dlst = [d]
    while(len(q) > 0):
      start_term = q.popleft()
      d = dlst.pop(0)
      # start_term is visited
      if len(chiln := self.children[start_term]) > 0:
        chiln = [x for x in chiln if not self.visited[x]]
        dlst += [d+1] * len(chiln)
        self.term_depth.update({x: d+1 for x in chiln})
        q.extend(chiln)
      self.visited[start_term] = True

  def precompute(self):
    self.MAX_BITS = len(bin(self.num_terms)) - 2
    self.ancestors = [[-1] * self.MAX_BITS for _ in range(self.num_terms)]

    for node in range(self.num_terms):
      self.ancestors[node][0] = self.parents[node]

    for bit in range(1, self.MAX_BITS):
        for node in range(self.num_terms):
            if self.ancestors[node][bit - 1] != -1:
                self.ancestors[node][bit] = self.ancestors[self.ancestors[node][bit - 1]][bit - 1]

  def find_lca(self, node1, node2):
      if self.depth_lst[node1] > self.depth_lst[node2]:
          node1, node2 = node2, node1

      # Lift node2 to the same level as node1
      for bit in range(self.MAX_BITS - 1, -1, -1):
          if self.depth_lst[node2] - (1 << bit) >= self.depth_lst[node1]:
              node2 = self.ancestors[node2][bit]

      if node1 == node2:
          return node1

      # Lift both nodes until they have the same parent
      for bit in range(self.MAX_BITS - 1, -1, -1):
          if self.ancestors[node1][bit] != self.ancestors[node2][bit]:
              node1 = self.ancestors[node1][bit]
              node2 = self.ancestors[node2][bit]

      return self.ancestors[node1][0]

  def build_lca_table(self):
    self.mica_table = [[-1] * self.num_terms for _ in range(self.num_terms)]

    for node1 in range(self.num_terms):
      for node2 in range(node1 + 1, self.num_terms):
        lca = self.find_lca(node1, node2)
        self.mica_table[node1][node2] = lca
        self.mica_table[node2][node1] = lca

  def find_mica(self, node1: str, node2: str):
    return self.mica_table[self.term_idx[node1]][self.term_idx[node2]]
  
  def get_resnik(self, node1: str, node2: str):
    return self.term_ic[self.term_lst[x]] \
           if (x := self.find_mica(node1, node2)) != -1 \
           else 0

  def lin_sim(self, node1: str, node2: str):
    return self.get_resnik(node1, node2) / (self.term_ic[node1] + self.term_ic[node2])

class SsLin(object):
  def __init__(self, onto: op.Ontology,
               term_lst: T.List[str],
               ic_ary: np.ndarray,):
    self.ont = onto.ont
    self.term_lst = term_lst
    self.term_set = set(term_lst)
    self.num_terms = len(term_lst)
    self.term_idx = {x: i for i, x in enumerate(term_lst)}
    self.ic_ary = ic_ary

    self.parents = [[self.term_idx[x] for x in ts]
                    if len(ts := self.term_set.intersection(
                                  self.ont[self.term_lst[i]]["is_a"])) > 0
                    else [-1]
                    for i in range(self.num_terms)]

    self.children: T.Dict[int, T.List[int]] = {i: [] for i in range(self.num_terms)}
    for i in range(self.num_terms):
      for x in self.parents[i]:
        if x != -1:
          self.children[x].append(i)
    
    # get the ancestors of each term
    self.ancestors: T.List[T.Set[int]] = [set() for _ in range(self.num_terms)]

    # call precompute for calculating the ancestors of each term
    self.precompute()

    # the initial value of the most informative common ancestor (MICA) table is -1
    self.mica_table = np.identity(self.num_terms, dtype=np.int64) * -1

    # call the calculate_mica_table function to calculate the MICA table
    self.build_mica_table()

  # Topological Sorting using Kahn's algorithm
  def topological_sort(self):
    in_degree = {node: 0 for node in range(self.num_terms)}
    for node in range(self.num_terms):
      for child in self.children[node]:
        in_degree[child] += 1
    
    queue = [node for node in range(self.num_terms) if in_degree[node] == 0]

    while queue:
      node = queue.pop(0)
      yield node
      for child in self.children[node]:
        in_degree[child] -= 1
        if in_degree[child] == 0:
          queue.append(child)
  
  def precompute(self):
    # Perform the DFS using topological sorting
    # to get the ancestors of each term
    for node in self.topological_sort():
      if self.children[node]:
        for child in self.children[node]:
          self.ancestors[child].add(node)
          self.ancestors[child].update(self.ancestors[node])
  
  def build_mica_table(self):
    # Calculate the most informative common ancestor (MICA) for each pair of nodes (t1, t2) in the graph.
    # the MICA has the highest information content among all the common ancestors of t1 and t2.
    for i in range(self.num_terms):
      for j in range(i+1, self.num_terms):
        common_ancestors = list(self.ancestors[i].intersection(self.ancestors[j]))
        if common_ancestors:
          self.mica_table[i,j] = max(common_ancestors,
                                     key=lambda x: self.ic_ary[x])
          self.mica_table[j,i] = self.mica_table[i,j]

def lin_similarity(term_idx: T.Dict[str, int],
                   ic_ary: np.ndarray,
                   mica_table: np.ndarray,
                   proteins1: T.List[str], proteins2: T.List[str]):
  m = len(proteins1)
  n = len(proteins2)
  annots1 = [term_idx[x] for x in proteins1]
  annots2 = [term_idx[x] for x in proteins2]
  sub_mica = mica_table[np.ix_(annots1, annots2)]
  sub_ic1 = ic_ary[annots1]
  sub_ic2 = ic_ary[annots2]
  zic = np.zeros((m,n), dtype=np.float32)
  index = np.where(sub_mica != -1)
  zic[index] = ic_ary[sub_mica[index]]
  score_mat = zic * 2 / (sub_ic1[:, None] + sub_ic2)
  return (score_mat.max(1).sum() + score_mat.max(0).sum()) / (m+n)

class SsWang(object):
  def __init__(self, ont: op.Ontology,
               term_lst: T.List[str]) -> None:
    self.ont = ont.ont
    self.term_lst = term_lst
    self.term_set = set(term_lst)
    self.num_terms = len(term_lst)
    self.term_idx = {x: i for i, x in enumerate(term_lst)}

    self.parents = [[self.term_idx[x] for x in ts]
                    if len(ts := self.term_set.intersection(
                                  self.ont[self.term_lst[i]]["is_a"])) > 0
                    else [-1]
                    for i in range(self.num_terms)]

    self.children: T.Dict[int, T.List[int]] = {i: [] for i in range(self.num_terms)}
    for i in range(self.num_terms):
      for x in self.parents[i]:
        if x != -1:
          self.children[x].append(i)
    
    self.weight_p2c = [{x: 0.8 for x in self.children[i]} for i in range(self.num_terms)]

    # get the ancestors of each term
    self.ancestors: T.List[T.Set[int]] = [set() for _ in range(self.num_terms)]

    self.precompute() # precompute the ancestors of each term

    # the initial value of S-value is a diagonal matrix with 1s on the diagonal
    self.s_value = np.identity(self.num_terms, dtype=np.float32)
    self.calculate_s_value()

    # calculate the semantic value of each term
    # the semantic value of a term is the sum of the S-value of all its ancestors
    self.semantic_value = np.zeros(self.num_terms, dtype=np.float32)
    for node in range(self.num_terms):
      self.semantic_value[node] = np.sum(self.s_value[node, :])

  # Topological Sorting using Kahn's algorithm
  def topological_sort(self):
    in_degree = {node: 0 for node in range(self.num_terms)}
    for node in range(self.num_terms):
      for child in self.children[node]:
        in_degree[child] += 1
    
    queue = [node for node in range(self.num_terms) if in_degree[node] == 0]

    while queue:
      node = queue.pop(0)
      yield node
      for child in self.children[node]:
        in_degree[child] -= 1
        if in_degree[child] == 0:
          queue.append(child)
  
  def precompute(self):
    # Perform the DFS using topological sorting
    # to get the ancestors of each term
    for node in self.topological_sort():
      if self.children[node]:
        for child in self.children[node]:
          self.ancestors[child].add(node)
          self.ancestors[child].update(self.ancestors[node])

  
  def calculate_s_value(self):
    # Calculate the S-value for each pair of nodes (A, t) in the graph, denoted as S_A(t).
    # The definition of S_A(t) is as follows:
    # {S_A(A) = 1, S_A(t) = max{w_e * S_A(t') | t' is a child of t}, if t is not equal to A}
    # the calculation of S_A(t) is to find the best path from A to t
    # where the product of the weights of all edges on the path is the largest.
    # note when t is not the ancestor of A, S_A(t) = 0

    # Therefore, this is alos a single-source maximal path solving process for each node
    # in the graph.
    # we will use the dynamic programming to solve this problem (loop through the topological order)
    topological_order = list(self.topological_sort())
    visited = [False] * self.num_terms
    for node in topological_order:
      if visited[node]: continue
      q = [node]
      s_values = np.array([-1] * self.num_terms, dtype=np.float32)
      s_values[node] = 1
      while q:
        curr = q.pop(0)
        visited[curr] = True
        for child in self.children[curr]:
          if not visited[child]:
            q.append(child)
          s_values[child] = max(s_values[child], self.weight_p2c[curr][child] * s_values[curr])
      
      # the s-value of the node that not be traversed is 0
      s_values[s_values == -1] = 0
      # ont of the ancestor of all the traversed nodes is the current node
      self.s_value[:, node] = s_values # update the S-value matrix
      # when we calculate S_A(t), it means self.s_value[A, t]
  
  def wang_sim(self, A: str | int, B: str | int):
    # sim(Wang)(A, B) = sum_{ S_A(t) + S_B(t)  | t belong to T_A and T_B} / (S_V(A) + S_V(B))
    # where S_V denotes the semantics value of a term
    # S_A(t) denotes the S-value of term t that related to term A
    # T_A denotes the ancestors of term A
    a = self.term_idx[A] if isinstance(A, str) else A
    b = self.term_idx[B] if isinstance(B, str) else B
    common_ancestors = list(self.ancestors[a].intersection(self.ancestors[b]))
    if common_ancestors:
      index = np.ix_([a,b], common_ancestors)
      sum_of_svalue = self.s_value[index].sum()
    else:
      sum_of_svalue = 0.
    return sum_of_svalue / \
            (self.semantic_value[a] + self.semantic_value[b])
  
  def build_wang_table(self):
    # vec_wang_sim = np.vectorize(self.wang_sim)
    # self.wang_table = vec_wang_sim(np.arange(self.num_terms)[:, None],
    #                                 np.arange(self.num_terms)[None, :])
    self.wang_table = np.identity(self.num_terms, dtype=np.float32)
    for i in range(self.num_terms):
      for j in range(i+1, self.num_terms):
        self.wang_table[i,j] = self.wang_sim(i, j)
        self.wang_table[j,i] = self.wang_table[i,j]


def wang_similarity(term_idx: T.Dict[str, int],
                    s_value: np.ndarray,
                    semantic_value: np.ndarray,
                    ancestors: T.List[T.Set[int]],
                    proteins1: T.List[str], proteins2: T.List[str]):
  m, n = len(proteins1), len(proteins2)
  annots1 = np.array([term_idx[p] for p in proteins1], dtype=np.int64)
  annots2 = np.array([term_idx[p] for p in proteins2], dtype=np.int64)
  @np.vectorize
  def sum_of_pair(a: int, b: int):
    common_ancestor = list(ancestors[a].intersection(ancestors[b]))
    idx = np.ix_([a, b], common_ancestor)
    return s_value[idx].sum()

  # sum_of_svalues = np.array([[sum_of_pair(a, b) for b in annots2] for a in annots1])
  sum_of_svalues = sum_of_pair(annots1[:, None], annots2)
  sv_1 = semantic_value[annots1]
  sv_2 = semantic_value[annots2]
  sum_of_svalues /= (sv_1[:, None] + sv_2)
  # BMA strategy
  return (sum_of_svalues.max(1).sum() + sum_of_svalues.max(0).sum()) / (m + n)

if __name__ == "__main__":
  import argparse as argp
  
  parser = argp.ArgumentParser()
  parser.add_argument("obo", help="go.obo filepath")
  parser.add_argument("data", help="a pickle format file saved by pandas")
  parser.add_argument("saving_path")
  parser.add_argument("-f", "--saving-format", default="pickle", 
                      choices=["pickle", "json"])
  
  opt = parser.parse_args()

  ont = op.Ontology(opt.obo, with_rels=True)
  with open(opt.data, "rb") as h:
     data: pd.DataFrame = pickle.load(h)
  
  namespace_terms = building(ont, data)
  serialize.match_by(opt.saving_format, namespace_terms, opt.saving_path)