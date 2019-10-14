import torch
import torchtext
import h5py
from config import args
import os

vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
vocab.itos.extend(['<unk>'])
vocab.stoi['<unk>'] = len(vocab.vectors)
vocab.vectors = torch.cat([vocab.vectors,torch.zeros(1,vocab.dim)],dim=0)

temporal_signals = ['before','after','while','then']
temporal_labels = ['SBAR-TMP','PP-TMP']
clause_level_labels = ['ROOT','S','SBAR','SBARQ','SINV','SQ']
phrase_level_labels = ['ADJP','ADVP','CONJP','FRAG','INTJ','LST','NAC', 'NP', 'NX', 'PP',
                       'PRN','PRT','QP','RRC','UCP','VP','WHADJP','WHADVP','WHAVP','WHNP','WHPP','X']
word_level_labels = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS',
                     'PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG',
                     'VBN','VBP','VBZ','WDT','WP','WP$','WRB',]
punctuation_labels = [',',"''",'.','-LRB-','-RRB-','<STOP>','``',':','$']

bracket_labels = clause_level_labels+phrase_level_labels+word_level_labels

possible_segments = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                     (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                     (0, 2), (1, 3), (2, 4), (3, 5),
                     (0, 3), (1, 4), (2, 5),
                     (0, 4), (1, 5),
                     (0, 5)]

feat_root = args.feat_root
features_h5py = h5py.File(os.path.join(feat_root, "average_fc7.h5"), 'r')
features = {}
for key in features_h5py.keys():
    features[key] = features_h5py[key][:]
features_h5py.close()
rgb_features = features

features_h5py = h5py.File(os.path.join(feat_root, "average_global_flow.h5"), 'r')
features = {}
for key in features_h5py.keys():
    features[key] = features_h5py[key][:]
features_h5py.close()
flow_features = features

TEF = torch.Tensor(possible_segments) / 6
prop_num = TEF.shape[0]
conTEF = torch.cat([TEF[:, None].expand(-1, prop_num, -1),
                    TEF[None].expand(prop_num, -1, -1)], dim=2)
