import os
import json
import random
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchtext

from data_processing import possible_segments, rgb_features, flow_features, vocab
from cstlstm import tree_batch
import tree_reformating

random.seed(0)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class DetectionDataset(Dataset):
    def __init__(self,split):

        self.split = split

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class DiDeMo(DetectionDataset):
    def __init__(self, split,
                 json_path='/localdisk/szhang83/Developer/LocalizingMoments/data/{}_data.json',
                 cache_path='/localdisk/szhang83/Developer/NSGV-Similarity/cache/{}_data.pkl'):
        super(DiDeMo, self).__init__(split)
        self.vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()

        # Language Feat
        cache_path = cache_path.format(self.split)
        if os.path.exists(cache_path):
            self.data = pickle.load(open(cache_path, 'rb'))
        else:
            json_path = json_path.format(self.split)
            original_data = json.load(open(json_path, 'r'))
            self.data = self.preprocess(original_data)
            pickle.dump(self.data, open(cache_path,'wb'))
        params = {}

        # Visual Features
        self.rgb_features = rgb_features
        self.flow_features = flow_features

    def preprocess(self, data):
        import nltk
        nltk.download('punkt')
        import benepar
        benepar.download('benepar_en2_large')
        parser = benepar.Parser("benepar_en2_large")
        from tqdm import tqdm
        progress_bar = tqdm(total=len(data))
        for d in data:
            d['parse_tree'] = parser.parse(d['description'])
            if 'reference_description' in d:
                d['ref_parse_tree'] = parser.parse(d['reference_description']) if len(d['reference_description'])>0 else d['parse_tree']
            progress_bar.update(1)
        progress_bar.close()
        return data

    def get_testing_item(self, index):

        num_segments = self.data[index]['num_segments']
        times = self.data[index]['train_times'] if self.split == 'train' and 'train_times' in self.data[index] else self.data[index]['times']
        video = self.data[index]['video']
        parse_tree = self.data[index]['parse_tree']
        parse_tree = tree_reformating.reformat_tree(parse_tree)
        parse_tree = tree_batch.sent_to_tree(parse_tree)

        rgb_feature = torch.from_numpy(self.rgb_features[video]).float()
        rgb_feature = F.pad(rgb_feature, [0,0,0,6-rgb_feature.shape[0]], "constant", 0)
        flow_feature = torch.from_numpy(self.flow_features[video]).float()
        flow_feature = F.pad(flow_feature, [0,0,0,6-flow_feature.shape[0]], "constant", 0)

        if isinstance(times[0],list):
            rint = random.randint(0, len(times) - 1)
            gt_time = times[rint]
        else:
            gt_time = times
        gt = possible_segments.index(tuple(gt_time))

        assert gt_time[1] < num_segments, 'gt {} longer than num_segments {}'.format(gt_time, num_segments)

        vis_mask = torch.zeros((len(possible_segments),1))
        for i, (start, end) in enumerate(possible_segments):
            if end < num_segments:
                vis_mask[i] = 1

        return [parse_tree, rgb_feature, flow_feature, vis_mask, gt]


    def __getitem__(self, index):
        return self.get_testing_item(index)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        batch_tree, batch_rgb_feats, batch_flow_feats, batch_vis_mask, batch_gt = [b for b in zip(*batch)]

        batch_data = {
            'batch_tree': tree_batch.Forest(batch_tree),
            'batch_rgb_feats': torch.stack(batch_rgb_feats),
            'batch_flow_feats': torch.stack(batch_flow_feats),
            'batch_mask': torch.stack(batch_vis_mask),
            'batch_gt': batch_gt,
        }

        return batch_data

class TEMPO_HL(DiDeMo):
    def __init__(self, split,
                 json_path='/localdisk/szhang83/Developer/LocalizingMoments/data/tempoHL+didemo_{}.json',
                 cache_path='/localdisk/szhang83/Developer/NSGV-Stream/cache/tempoHL+didemo_{}_data.pkl'):
        super(TEMPO_HL, self).__init__(split, json_path=json_path, cache_path=cache_path)

    def get_testing_item(self, index):

        parse_tree, rgb_feature, flow_feature, vis_mask, gt = super(TEMPO_HL, self).get_testing_item(index)

        context = self.data[index]['context']

        if len(context) > 0:
            context_gt = possible_segments.index(tuple(context))
        else:
            num_segments = self.data[index]['num_segments']
            context_gt = -1 if num_segments == 6 else -3

        # if self.data[index]['annotation_id'].split('_')[0] in ['after','before','then','while']:#before_4390, , after_4273, 'after_4258'
        # if self.data[index]['annotation_id'] == 'after_4258':
        #     print(self.data[index]['description'])
        return [parse_tree, rgb_feature, flow_feature, vis_mask, gt, context_gt]

    def collate_fn(self, batch):

        batch_tree, batch_rgb_feats, batch_flow_feats, batch_vis_mask, batch_gt, batch_context_gt = [b for b in zip(*batch)]

        batch_data = {
            'batch_tree': tree_batch.Forest(batch_tree),
            'batch_rgb_feats': torch.stack(batch_rgb_feats),
            'batch_flow_feats': torch.stack(batch_flow_feats),
            'batch_mask': torch.stack(batch_vis_mask),
            'batch_gt': batch_gt,
            'batch_context_gt': batch_context_gt,
        }

        return batch_data

class TEMPO_TL(TEMPO_HL):
    def __init__(self, split,
                 json_path='/localdisk/szhang83/Developer/LocalizingMoments/data/tempoTL+didemo_{}.json',
                 cache_path='/localdisk/szhang83/Developer/NSGV-Stream/cache/tempoTL+didemo_{}_data.pkl'):
        super(TEMPO_TL, self).__init__(split, json_path=json_path, cache_path=cache_path)

if __name__ == '__main__':
    from config import args
    import sys
    dataset = getattr(sys.modules[__name__], args.dataset_name)(args.split)
    # for i in range(len(dataset)):
    #     dataset[i]
    # dataset.collate_fn([dataset[0], dataset[1], dataset[2]])
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            collate_fn=dataset.collate_fn)

    i = 0
    for batch in dataloader:
        print(i)
        i+=1