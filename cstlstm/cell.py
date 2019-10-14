"""Batch Child-Sum Tree-LSTM cell for parallel processing of nodes per level."""
import torch
import torch.nn as nn
import itertools
from data_processing import args

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=args.dropout_language)
        self.conv_combined = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.conv_wf = nn.Linear(input_size, hidden_size)
        self.conv_uf = nn.Linear(hidden_size, hidden_size)

        if args.block_orthogonal:
            self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.conv_combined.weight.data[:,:self.input_size], [self.hidden_size, self.input_size])
        block_orthogonal(self.conv_combined.weight.data[:,self.input_size:], [self.hidden_size, self.hidden_size])
        block_orthogonal(self.conv_wf.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.conv_uf.weight.data, [self.hidden_size, self.hidden_size])
        self.conv_combined.bias.data.fill_(0)
        self.conv_wf.bias.data.fill_(1.0)
        self.conv_uf.bias.data.fill_(1.0)

    def forward(self, inputs, previous_states):
        # prepare the inputs
        cell_states = previous_states[0]
        hidden_states = previous_states[1]

        inputs_mat = inputs
        h_tilde_mat = torch.cat([torch.sum(h, 0).expand(1, self.hidden_size)
                                 for h in hidden_states], dim=0)

        prev_c_mat = torch.cat(cell_states, 0)
        big_cat_in = torch.cat([inputs_mat, h_tilde_mat], 1)

        big_cat_out = self.conv_combined(big_cat_in)
        z_i, z_o, z_u = big_cat_out.split(self.hidden_size, 1)

        # apply dropout to u, like the Fold boys
        z_u = self.dropout(z_u)

        # forget gates
        f_inputs = self.conv_wf(inputs_mat)

        lens = [t.size()[0] for t in hidden_states]
        start = [sum([lens[j] for j in range(i)]) for i in range(len(lens))]
        end = [start[i] + lens[i] for i in range(len(lens))]

        # we can then go ahead and concatenate for matmul
        prev_h_mat = torch.cat(hidden_states, 0)
        f_hiddens = self.conv_uf(prev_h_mat)

        # compute the f_jks by expanding the inputs to the same number
        # of rows as there are prev_hs for each, then just do a simple add.
        f_inputs_split = f_inputs.split(1, 0)
        f_inputs_expanded = [f_inputs_split[i].expand(lens[i], self.hidden_size)
                             for i in range(len(lens))]
        f_inputs_ready = torch.cat(f_inputs_expanded, 0)
        f_jks = torch.sigmoid(f_inputs_ready + f_hiddens)

        # cell and hidden state
        fc_mul = f_jks * prev_c_mat
        split_fcs = [fc_mul[start[i]:end[i]] for i in range(len(lens))]
        fc_term = torch.cat([torch.sum(item, 0).expand(1, self.hidden_size)
                             for item in split_fcs])
        c = torch.sigmoid(z_i) * torch.tanh(z_u) + fc_term
        h = torch.sigmoid(z_o) * torch.tanh(c)

        assert torch.sum(torch.isnan(h)).item() == 0

        return c, h

def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """

    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]


