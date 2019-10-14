"""Tree data structures and functions for parallel processing."""
import numpy as np
import nltk
import torch
from data_processing import temporal_signals

def cumsum(seq):
    """Get the cumulative sum of a sequence of sequences at each index.
    Args:
      seq: List of sequences.
    Returns:
      List of integers.
    """
    r, s = [], 0
    for e in seq:
        l = len(e)
        r.append(l + s)
        s += l
    return r

def flatten_list_of_lists(list_of_lists):
    """Flatten a list of lists.
    Args:
      list_of_lists: List of Lists.
    Returns:
      List.
    """
    return [item for sub_list in list_of_lists for item in sub_list]

def get_adj_mat(nodes):
    """Get an adjacency matrix from a node set.
    A row in the matrix indicates the children of the node at that index.
    A column in the matrix indicates the parent of the node at that index.
    Args:
      nodes: List of Nodes.
    Returns:
      2D numpy.ndarray: an adjacency matrix.
    """
    size = len(nodes)
    mat = np.zeros((size, size), dtype='int32')
    for node in nodes:
        if node.parent_id >= 0:
            mat[node.parent_id][node.id] = 1
    return mat

def get_child_ixs(nodes, adj_mat):
    """Get lists of children indices at each level.
    We need this for batching, to show the wiring of the nodes at each level,
    as we process them in parallel.
    Args:
      nodes: Dictionary of {Integer: [List of Nodes]} for the nodes at each
        level in the tree / forest.
      adj_mat: 2D numpy.ndarray, adjacency matrix for all nodes.
    Returns:
      Dictionary of {Integer: [[List of child_ixs @ l+1] for parent_ixs @ l]}.
    """
    child_ixs = {}
    # We don't need child_ixs for the last level so just range(max_level) not +1
    for l in range(max(nodes.keys())):
        child_nodes = nodes[l+1]
        id_to_ix = {child_nodes[ix].id: ix for ix in
                    range(len(child_nodes))}
        ids = [np.nonzero(adj_mat[n.id])[0] for n in nodes[l]]
        try:
            ixs = [[id_to_ix[id] for id in id_list] for id_list in ids]
        except Exception as e:
            print('level: %s' % l)
            print('child_ixs state')
            print(child_ixs)
            print('child_nodes')
            print(child_nodes)
            print('id_to_ix')
            print(id_to_ix)
            raise e
        child_ixs[l] = ixs
    return child_ixs

def get_max_level(nodes):
    """Get the highest level number given a list of nodes.
    Args:
      nodes: List of Nodes.
    Returns:
      Integer, the highest level number. It is a zero-based number, so if later
        the actual number of levels is desired, will need to add one to this.
    """
    return max([n.level for n in nodes])

def get_nodes_at_levels(nodes):
    """Get a dictionary listing nodes at each level.
    Args:
      nodes: List of Nodes.
    Returns:
      Dictionary of {Integer: [List of Nodes]} for each level.
    """
    max_level = get_max_level(nodes)
    return dict(zip(
        range(max_level+1),
        [[n for n in nodes if n.level == l]
         for l in range(max_level+1)]))

def offset_node_lists(node_lists):
    """Offset the ids in the list of node lists.
    Args:
      node_lists: List of Lists of Nodes.
    Returns:
      List of Lists of Nodes.
    """
    cumsums = cumsum(node_lists)
    for list_ix in range(len(node_lists)):
        for node in node_lists[list_ix]:
            offset = cumsums[list_ix - 1] if list_ix > 0 else 0
            node.id = node.id + offset
            node.parent_id = node.parent_id + offset \
                if node.parent_id >= 0 \
                else -1
    return node_lists



# Model Classes

class Forest:
    def __init__(self, trees):
        """Create a new Forest.
        Args:
          trees: List of Trees. They will be processed in order. Pass them in
            the desired order.
        """
        self.trees = trees
        node_lists = offset_node_lists([tree.node_list for tree in trees])
        self.node_list = flatten_list_of_lists(node_lists)
        self.max_level = get_max_level(self.node_list)
        self.nodes = get_nodes_at_levels(self.node_list)
        self.size = len(self.node_list)
        self.max_level = get_max_level(self.node_list)
        self.adj_mat = get_adj_mat(self.node_list)
        self.child_ixs = get_child_ixs(self.nodes, self.adj_mat)

class Node:

    def __init__(self, tag, token, id, parent_id, level, is_leaf):
        """Create a new Node."""
        self.tag = tag
        self.token = token
        self.id = id
        self.parent_id = parent_id
        self.level = level
        self.is_leaf = is_leaf

class Tree:

    def __init__(self, nodes):
        """Create a new Tree.
        Args:
          nodes: List of Nodes.
        """
        self.node_list = nodes
        self.nodes = get_nodes_at_levels(self.node_list)
        self.size = len(self.node_list)
        self.max_level = get_max_level(self.node_list)
        self.adj_mat = get_adj_mat(self.node_list)
        self.child_ixs = get_child_ixs(self.nodes, self.adj_mat)

class Queue:
    def __init__(self):
        self.data = []

    def empty(self):
        return len(self.data) == 0

    def push(self, token, level):
        self.data.append((token, level))

    def pop(self):
        token, level = self.data[0]
        del self.data[0]
        return token, level

class Stack:

    def __init__(self):
        self.items = []

def sent_to_tree(sent, batch_idx=None):
    nodes = []
    q = Queue()
    id = 0
    sent.id = id
    sent.parent_id = -1
    sent.parent_tag = 'ROOT'

    q.push(sent, 0)
    while not q.empty():
        token, level = q.pop()
        node = token_to_node(token, level)
        nodes.append(node)
        for child in filter(lambda x: isinstance(x, nltk.Tree), token):
            id += 1
            child.id, child.parent_id, child.parent_tag = id, token.id, node.tag
            q.push(child, level + 1)
    return Tree(nodes)

from data_processing import vocab
def token_to_node(token, level):

    return Node(
        tag = token.label() if token.id != 0 else 'ROOT',
        token = torch.LongTensor([vocab.stoi.get(w.lower(),400000) for w in token.leaves()]),
        id = token.id,
        parent_id = token.parent_id,
        level = level,
        is_leaf=token.height()==2,
    )