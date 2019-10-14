import torch
from torch import nn
from . import cell, prev_states
from .text_embedding import TextEmbedding

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChildSumTreeLSTM, self).__init__()


        self.tree_cell = cell.ChildSumTreeLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        # Initialize previous states (to get wirings from nodes on lower level)
        self._prev_states = prev_states.PreviousStates(hidden_size)

    def forward(self, forest, inputs):
        outputs = {}
        # Work backwards through level indices - i.e. bottom up.
        for l in reversed(range(forest.max_level + 1)):
            # Get input word vectors for this level.

            # Get previous hidden states for this level.
            if l == forest.max_level:
                hidden_states = self._prev_states.zero_level(
                    len(forest.nodes[l]))
            else:
                hidden_states = self._prev_states(
                    level_nodes=forest.nodes[l],
                    level_up_wirings=forest.child_ixs[l],
                    prev_outputs=outputs[l+1])

            outputs[l] = self.tree_cell(inputs[l], hidden_states)

        return outputs