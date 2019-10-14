import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-feature_type_0', default='rgb', choices=['flow','rgb'], type=str)
parser.add_argument('-feature_type_1', default='flow', choices=['flow','rgb'], type=str)
parser.add_argument('-dataset_name', default='TEMPO_HL', choices=['TEMPO_HL','TEMPO_TL'], type=str)
parser.add_argument('-batch_size', default=1, type=int)
parser.add_argument('-lr', default=0.001, type=int)
parser.add_argument('-dropout_visual', default=0.0, type=float)
parser.add_argument('-dropout_language', default=0.0, type=float)
parser.add_argument('-context_weight', default=1.0, type=float)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-num_workers', default=0, type=int)
parser.add_argument('-block_orthogonal', default=False, action='store_true')
parser.add_argument('-strong_supervised', default=False, action='store_true')
parser.add_argument('-vis_hidden_size', default=500, type=int)
parser.add_argument('-lang_hidden_size', default=1000, type=int)
parser.add_argument('-att_hidden_size', default=250, type=int)
parser.add_argument('-hidden_size', default=250, type=int)
parser.add_argument('-max_epoch', default=20, type=int)
parser.add_argument('-test_interval', default=1, type=float)
parser.add_argument('-split', default='val', choices=['train','val','test'], type=str)
parser.add_argument('-verbose', default=False, action='store_true')
parser.add_argument('-feat_root', default='./data/', type=str)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_prefix = './'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu