import torch
from torch import nn
from data_processing import conTEF
from .module import MCN, PairLocationModule, NodeAttentionModule, PropAttentionModule
from cstlstm.text_embedding import TextEmbedding
from cstlstm.encoder import ChildSumTreeLSTM


class MLLC_Tree(nn.Module):

    def __init__(self, input_size_0, input_size_1, txt_input_size, hidden_size):
        super(MLLC_Tree, self).__init__()
        self.mcn = MCN()
        self.txt_input_size = txt_input_size
        self.text_embedding = TextEmbedding()
        self.cstlstm = ChildSumTreeLSTM(self.text_embedding.dim, txt_input_size)
        self.prop_score_builder = PairLocationModule(4+input_size_0+input_size_1,txt_input_size,hidden_size)
        self.conTEF = nn.Parameter(conTEF,requires_grad=False)

    def forward(self, forest, input_0, input_1, visual_mask):

        prop_feat_0 = self.mcn(input_0, visual_mask)
        prop_feat_1 = self.mcn(input_1, visual_mask)
        batch_size, prop_num, _  =  prop_feat_0.shape

        inputs = {}
        for l in reversed(range(forest.max_level + 1)):
            word_vectors, word_mask = self.text_embedding(forest.nodes[l])

            inputs[l] = torch.cat([(word_vectors[i, None, 0] if n.is_leaf
                                    else word_vectors.new_zeros(1, word_vectors.shape[-1]))
                                   for i, n in enumerate(forest.nodes[l])],dim=0)
        tree_out = self.cstlstm(forest, inputs)

        cat_prop_feat = torch.cat([prop_feat_0[:,:,None].expand(-1,-1,prop_num,-1),
                                       prop_feat_1[:,None].expand(-1,prop_num,-1,-1),
                                       self.conTEF[None].expand(batch_size,-1,-1,-1)],dim=3)

        scores = self.prop_score_builder(tree_out[0][1],cat_prop_feat)

        return scores

class MLLC_Signal(MLLC_Tree):

    def __init__(self, input_size_0, input_size_1, txt_input_size, hidden_size):
        super(MLLC_Signal, self).__init__(input_size_0, input_size_1, txt_input_size, hidden_size)
        self.loc_score_builder = PairLocationModule(4,txt_input_size,hidden_size)
        self.signal_builder = NodeAttentionModule(txt_input_size)

    def forward(self, forest, input_0, input_1, visual_mask):

        prop_feat_0 = self.mcn(input_0, visual_mask)
        prop_feat_1 = self.mcn(input_1, visual_mask)
        batch_size, prop_num, _  =  prop_feat_0.shape

        inputs = {}
        for l in reversed(range(forest.max_level + 1)):
            word_vectors, word_mask = self.text_embedding(forest.nodes[l])

            inputs[l] = torch.cat([(word_vectors[i, None, 0] if n.is_leaf
                                    else word_vectors.new_zeros(1, word_vectors.shape[-1]))
                                   for i, n in enumerate(forest.nodes[l])],dim=0)
        tree_out = self.cstlstm(forest, inputs)

        cat_prop_feat = torch.cat([prop_feat_0[:,:,None].expand(-1,-1,prop_num,-1),
                                       prop_feat_1[:,None].expand(-1,prop_num,-1,-1),
                                       self.conTEF[None].expand(batch_size,-1,-1,-1)],dim=3)

        prop_scores = self.prop_score_builder(tree_out[0][1],cat_prop_feat)

        signal_word_feat = self.signal_builder(forest, tree_out)
        loc_scores = self.loc_score_builder(signal_word_feat,self.conTEF[None].expand(batch_size,-1,-1,-1))

        scores = prop_scores+loc_scores
        return scores

class MLLC_FullAtt(MLLC_Signal):

    def __init__(self, input_size_0, input_size_1, txt_input_size, hidden_size):
        super(MLLC_FullAtt, self).__init__(input_size_0, input_size_1, txt_input_size, hidden_size)
        self.main_sent_builder = NodeAttentionModule(txt_input_size)
        self.context_sent_builder = NodeAttentionModule(txt_input_size)
        self.main_prop_att_builder = PropAttentionModule(input_size_0, txt_input_size, hidden_size)
        self.context_prop_att_builder = PropAttentionModule(input_size_1, txt_input_size, hidden_size)

    def forward(self, forest, input_0, input_1, visual_mask):

        prop_feat_0 = self.mcn(input_0, visual_mask)
        prop_feat_1 = self.mcn(input_1, visual_mask)
        batch_size, prop_num, _  =  prop_feat_0.shape

        inputs = {}
        for l in reversed(range(forest.max_level + 1)):
            word_vectors, word_mask = self.text_embedding(forest.nodes[l])

            inputs[l] = torch.cat([(word_vectors[i, None, 0] if n.is_leaf
                                    else word_vectors.new_zeros(1, word_vectors.shape[-1]))
                                   for i, n in enumerate(forest.nodes[l])],dim=0)
        tree_out = self.cstlstm(forest, inputs)

        signal_word_feat = self.signal_builder(forest, tree_out, judge=True)
        main_root_feat = self.main_sent_builder(forest, tree_out)
        context_root_feat = self.context_sent_builder(forest, tree_out)

        main_prop_input = self.main_prop_att_builder(main_root_feat, prop_feat_0, visual_mask)
        context_prop_input = self.context_prop_att_builder(context_root_feat, prop_feat_1, visual_mask)

        cat_prop_feat = torch.cat([main_prop_input[:,:,None].expand(-1,-1,prop_num,-1),
                                   context_prop_input[:,None].expand(-1,prop_num,-1,-1),
                                   self.conTEF[None].expand(batch_size,-1,-1,-1)],dim=3)

        prop_scores = self.prop_score_builder(tree_out[0][1],cat_prop_feat)

        loc_scores = self.loc_score_builder(signal_word_feat,self.conTEF[None].expand(batch_size,-1,-1,-1))

        scores = prop_scores+loc_scores
        return scores