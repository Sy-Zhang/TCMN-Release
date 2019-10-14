import torch
from torch import nn
import torchtext
from data_processing import vocab, bracket_labels, punctuation_labels
import numpy as np
from config import args
from config import device

class TextEmbedding(nn.Module):
    def __init__(self,with_pos=False, with_label=False):
        super(TextEmbedding, self).__init__()
        self.with_pos = with_pos
        self.with_label = with_label
        self.word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

        if with_label:
            self.label_embedding = nn.Embedding(len(bracket_labels+punctuation_labels), vocab.dim)
        if with_pos:
            self.position_embedding = torch.nn.Embedding.from_pretrained(
                self.position_encoding_init(n_position=args.max_language_length,emb_dim=vocab.dim))

        self.dim = 2*vocab.dim if with_label else vocab.dim

    def forward(self, nodes):
        # Word Embedding
        word_idxs = nn.utils.rnn.pad_sequence([node.token for node in nodes], batch_first=True, padding_value=400000).to(device)
        word_vectors = self.word_embedding(word_idxs)
        word_mask = nn.utils.rnn.pad_sequence([torch.ones(len(node.token),1) for node in nodes], batch_first=True).to(device)

        if self.with_pos:
            pos_idxs = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(list(range(1,len(node.token)+1))) for node in nodes],
                batch_first=True, padding_value=0).to(device)
            pos_vectors = self.position_embedding(pos_idxs)
            word_vectors = pos_vectors+word_vectors

        # Label Embedding
        if self.with_label:
            labels = torch.LongTensor([(bracket_labels+punctuation_labels).index(node.tag) for node in nodes]).to(device)
            label_vectors = self.label_embedding(labels)
            word_vectors = torch.cat([word_vectors, label_vectors], dim=1)

        return word_vectors, word_mask

    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)