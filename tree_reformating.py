from nltk.tree import Tree, ParentedTree
from data_processing import punctuation_labels

temporal_signals = ['before','while','then','after','until']

def remove_punctuation(parse_tree):
    for tree in parse_tree.subtrees(
            lambda t: t.height() == 2 and t.leaves()[0] in punctuation_labels):
        while tree.parent() is not None and len(tree.parent()) == 1:
            tree = tree.parent()
        tree.parent().remove(tree)
    return parse_tree

def is_tmp_tree(parse_tree):
    if parse_tree[0].leaves()[0] in temporal_signals:
        return True
    else:
        return False


def find_tmp_tree(parse_tree):
    tmp_tree = next(parse_tree.subtrees(lambda t: is_tmp_tree(t)), None)
    if tmp_tree is None:
        return []
    else:
        sub_tmp_tree = find_tmp_tree(tmp_tree)
        tmp_tree.parent().remove()

def lower_cases(parse_tree):
    for tree in parse_tree.subtrees(lambda t: t.height() == 2):
        tree[0] = tree[0].lower()
    return parse_tree

def reformat_tree(parse_tree):
    parse_tree = ParentedTree.fromstring(str(parse_tree))
    parse_tree = lower_cases(parse_tree)
    # reformated_tree = remove_punctuation(parse_tree)
    return parse_tree
