import torch
from torch import nn
import torch.nn.functional as F
from cstlstm.text_embedding import TextEmbedding
from data_processing import bracket_labels, punctuation_labels, vocab
from config import device, args

class MCN(nn.Module):
    def __init__(self):
        super(MCN, self).__init__()

    def forward(self, visual_input, visual_mask):
        proposal_encoding = []
        proposal_encoding.append(visual_input)

        vis_h = F.avg_pool1d(visual_input.transpose(1,2),2,1).transpose(1,2)
        proposal_encoding.append(vis_h)

        vis_h = F.avg_pool1d(visual_input.transpose(1,2),3,1).transpose(1,2)
        proposal_encoding.append(vis_h)

        vis_h = F.avg_pool1d(visual_input.transpose(1,2),4,1).transpose(1,2)
        proposal_encoding.append(vis_h)

        vis_h = F.avg_pool1d(visual_input.transpose(1,2),5,1).transpose(1,2)
        proposal_encoding.append(vis_h)

        vis_h = F.avg_pool1d(visual_input.transpose(1,2),6,1).transpose(1,2)
        proposal_encoding.append(vis_h)

        proposal_encoding = torch.cat(proposal_encoding, dim=1)
        proposal_encoding = F.normalize(proposal_encoding,dim=2)*visual_mask
        return proposal_encoding

class PairLocationModule(nn.Module):
    def __init__(self, vis_input_size, txt_input_size, hidden_size):
        super(PairLocationModule, self).__init__()
        self.visual_to_hidden = nn.Sequential(
            nn.Linear(vis_input_size, hidden_size),
        )
        self.textual_to_hidden = nn.Sequential(
            nn.Linear(txt_input_size, hidden_size),
        )
        self.vis_dropout = nn.Dropout(p=args.dropout_visual)

        # NOTE: FOR TEMPO-TL BEST
        # self.prediction = nn.Sequential(
        #     nn.Linear(hidden_size,hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size,1)
        # )

        self.nonlinear_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fuse_dropout = nn.Dropout(p=args.dropout_visual)
        self.nonlinear_2 = nn.Linear(hidden_size,1)

    def forward(self, textual_input, visual_input):
        vis_h = self.visual_to_hidden(visual_input)
        vis_h = self.vis_dropout(vis_h)
        txt_h = self.textual_to_hidden(textual_input)
        fused_h = F.normalize(txt_h[:,None,None].expand_as(vis_h)*vis_h,dim=3)

        # NOTE: FOR TEMPO-TL BEST
        # scores = self.prediction(fused_h).squeeze(-1)

        intermediate = self.nonlinear_1(fused_h)
        intermediate = self.fuse_dropout(intermediate)
        scores = self.nonlinear_2(intermediate).squeeze(-1)
        return scores

class NodeAttentionModule(nn.Module):
    def __init__(self, txt_input_size):
        super(NodeAttentionModule, self).__init__()

        self.text_embedding = TextEmbedding()
        self.label_embedding = nn.Embedding(len(bracket_labels + punctuation_labels), self.text_embedding.dim)
        self.node_attention_linear = nn.Sequential(
            nn.Linear(self.text_embedding.dim+txt_input_size, 1),
        )

    def forward(self, forest, node_embedding, judge=False):

        node_scores = {}
        for l in reversed(range(forest.max_level + 1)):
            labels = torch.LongTensor(
                [(bracket_labels + punctuation_labels).index(node.tag) for node in forest.nodes[l]]).to(device)
            label_input = self.label_embedding(labels)
            node_input = torch.cat([label_input,node_embedding[l][1]],dim=1)
            node_score = self.node_attention_linear(node_input)
            node_scores[l] = node_score

        batch_size = len(forest.trees)
        clause_scores = [[] for _ in range(batch_size)]
        clause_out = [[] for _ in range(batch_size)]
        clause_nodes = [[] for _ in range(batch_size)]
        for l in reversed(range(0, forest.max_level + 1)):
            offset = 0
            for j, tree in enumerate(forest.trees):
                for k, node in enumerate(tree.nodes.get(l, [])):
                    # if node.tag not in punctuation_labels:
                    clause_scores[j].append(node_scores[l][offset + k])
                    clause_out[j].append(node_embedding[l][1][offset + k])
                    clause_nodes[j].append([vocab.itos[i] for i in node.token])
                offset += len(tree.nodes.get(l, []))

        padded_clause_scores = nn.utils.rnn.pad_sequence([torch.cat(scores, dim=0) for scores in clause_scores],batch_first=True)
        padded_clause_mask = nn.utils.rnn.pad_sequence([torch.ones(len(scores)) for scores in clause_scores],batch_first=True)
        # Softmax
        clause_scores = F.softmax(padded_clause_scores, dim=1) * padded_clause_mask.to(device)
        clause_scores = clause_scores / torch.sum(clause_scores, dim=1, keepdim=True)

        # Weighted Context_Score
        if judge and clause_nodes[0][torch.argsort(clause_scores[0],descending=True)[0]] in [['before'],['after'],['while'],['then']]:
            print(clause_nodes[0][torch.argsort(clause_scores[0],descending=True)[0]])
        padded_clause_feat = nn.utils.rnn.pad_sequence([torch.stack(out) for out in clause_out], batch_first=True)
        context_root_feat = clause_scores[:, :, None] * padded_clause_feat
        context_root_feat = torch.sum(context_root_feat,dim=1)
        return context_root_feat

class PropAttentionModule(nn.Module):
    def __init__(self, vis_input_size, txt_hidden_size, hidden_size):
        super(PropAttentionModule, self).__init__()
        self.visual_to_hidden = nn.Sequential(
            nn.Linear(vis_input_size, args.vis_hidden_size),
            nn.ReLU(),
            nn.Linear(args.vis_hidden_size, args.att_hidden_size)
        )
        self.textual_to_hidden = nn.Sequential(
            nn.Linear(txt_hidden_size, args.att_hidden_size),
        )

        self.prediction = nn.Sequential(
            nn.Linear(args.att_hidden_size, args.att_hidden_size),
            nn.ReLU(),
            nn.Linear(args.att_hidden_size,1)
        )

    def forward(self, textual_input, visual_input, visual_mask):

        main_txt_h = self.textual_to_hidden(textual_input)
        main_vis_h = self.visual_to_hidden(visual_input)
        fused_h = F.normalize(main_txt_h[:, None].expand_as(main_vis_h) * main_vis_h, dim=2)
        scores = self.prediction(fused_h).squeeze(-1)
        scores = F.softmax(scores, dim=1) * visual_mask.squeeze(-1)
        scores = scores / torch.sum(scores, dim=1, keepdim=True)
        weighted_feat = scores[:, :, None] * visual_input
        return weighted_feat