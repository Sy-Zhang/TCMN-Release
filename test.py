import os
import pickle as pkl

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing import possible_segments
import dataset
from engine import Engine
from model.model import MLLC_FullAtt
from eval import eval_predictions

from config import device, args, path_prefix


def save_scores(scores, data, dataset_name, split):
    results = {}
    for i, d in enumerate(data):
        results[d['annotation_id']] = scores[i]
    pkl.dump(results,open(path_prefix+'results/{}_{}_{}-{}.pkl'.format(
        dataset_name, split, args.feature_type_0, args.feature_type_1),'wb'))

def test_model(split):
    verbose = args.verbose
    feature_type_0 = args.feature_type_0
    feature_type_1 = args.feature_type_1
    dataset_name = args.dataset_name
    model_path = path_prefix + 'checkpoint/{}/{}-{}-H{}/'.format(
        dataset_name, args.feature_type_0, args.feature_type_1, args.hidden_size)

    if feature_type_0 == 'rgb' and feature_type_1 == 'rgb':
        if dataset_name == "DiDeMo":
            model_path = os.path.join(model_path,'')
        elif dataset_name == "TEMPO_HL":
            model_path = os.path.join(model_path, 'iter009435-0.1959-0.4018.pkl')
        elif dataset_name == "TEMPO_TL":
                model_path = os.path.join(model_path,'iter005055-0.2793-0.3923.pkl')

    elif feature_type_0 == 'rgb' and feature_type_1 == 'flow':
        if dataset_name == "DiDeMo":
            model_path = os.path.join(model_path,'')
        elif dataset_name == "TEMPO_HL":
            model_path = os.path.join(model_path,'iter015725-0.2073-0.4349.pkl')
        elif dataset_name == "TEMPO_TL":
            model_path = os.path.join(model_path,'iter005055-0.2863-0.4019.pkl')

    elif feature_type_0 == 'flow' and feature_type_1 == 'rgb':
        if dataset_name == "DiDeMo":
            model_path = os.path.join(model_path,'')
        elif dataset_name == "TEMPO_HL":
            model_path = os.path.join(model_path,'iter018870-0.2235-0.4526.pkl')
        elif dataset_name == "TEMPO_TL":
            model_path = os.path.join(model_path,'iter010110-0.3055-0.4258.pkl')

    elif feature_type_0 == 'flow' and feature_type_1 == 'flow':
        if dataset_name == "DiDeMo":
            model_path = os.path.join(model_path,'')
        elif dataset_name == "TEMPO_HL":
            model_path = os.path.join(model_path,'iter012580-0.2403-0.4581.pkl')
        elif dataset_name == "TEMPO_TL":
            model_path = os.path.join(model_path,'iter015165-0.3341-0.4647.pkl')



    input_size_0 = 4096 if args.feature_type_0 == 'rgb' else 1024
    input_size_1 = 4096 if args.feature_type_1 == 'rgb' else 1024

    model = MLLC_FullAtt(input_size_0=input_size_0, input_size_1=input_size_1, txt_input_size=args.lang_hidden_size, hidden_size=args.hidden_size).to(device)

    model.eval()
    model_checkpoint = torch.load(model_path)
    model.load_state_dict(model_checkpoint)

    test_dataset = getattr(dataset, dataset_name)(split)

    dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=test_dataset.collate_fn)

    def network(sample):

        parse_tree = sample['batch_tree']
        input_0 = sample['batch_{}_feats'.format(args.feature_type_0)].to(device)
        input_1 = sample['batch_{}_feats'.format(args.feature_type_1)].to(device)
        visual_mask = sample['batch_mask'].to(device)
        batch_gt = sample['batch_gt']
        batch_context_gt = sample['batch_context_gt']
        output = model(parse_tree, input_0, input_1, visual_mask)

        batch_size, prop_num, _ = output.shape

        def ranking_loss(output, strong_supervised=False):
            loss_mask = visual_mask.view(batch_size, -1)
            main_output = torch.max(output,dim=2)[0]
            gt_predicted_score = torch.gather(main_output, 1, torch.LongTensor(batch_gt)[:,None].expand(-1,prop_num).to(device))

            loss = F.margin_ranking_loss(main_output.view(batch_size,-1),
                                         gt_predicted_score.view(batch_size,-1),
                                         -torch.ones(1).to(device),
                                         margin=0.1, reduction='none')*loss_mask
            ranking_loss = torch.sum(loss)/torch.sum(loss_mask)

            if strong_supervised:
                assert output.dim() == 3
                gt_predicted_score = []
                context_output = []
                loss_mask = visual_mask.view(batch_size, -1)
                for i, (gt,context_gt) in enumerate(zip(batch_gt,batch_context_gt)):
                    row = torch.stack([output[i, gt, context_gt] for _ in range(prop_num)])
                    context_output.append(output[i, gt])
                    gt_predicted_score.append(row)
                gt_predicted_score = torch.stack(gt_predicted_score).to(device)
                context_output = torch.stack(context_output).to(device)

                loss = F.margin_ranking_loss(context_output.view(batch_size, -1),
                                             gt_predicted_score.view(batch_size, -1),
                                             -torch.ones(1).to(device),
                                             margin=0.1, reduction='none') * loss_mask

                ranking_loss += args.context_weight*torch.sum(loss)/torch.sum(loss_mask)

            return ranking_loss


        # loss_value = 1*ranking_loss(rgb_output)+1*ranking_loss(flow_output)#+1*ranking_loss(loc_output)
        loss_value = ranking_loss(output,args.strong_supervised)

        assert torch.sum(torch.isnan(output)).item() == 0

        score_mask = visual_mask.expand(-1,-1,prop_num)*visual_mask.transpose(1,2).expand(-1,prop_num,-1)
        scores = (torch.max((10000+output)*score_mask,dim=2)[0])
        # TODO: fix zero and minus score problem

        return loss_value, scores

    # save to json file
    def on_test_start(state):
        state['sorted_segments_list'] = []
        state['scores'] = []
        if verbose:
            state['progress_bar'] = tqdm(total=len(test_dataset))

    def on_test_forward(state):
        scores = state['output'].cpu().data.numpy().squeeze()
        state['scores'].append(scores)
        sorted_index = np.argsort(scores)[::-1]
        sorted_segments = [possible_segments[i] for i in sorted_index]
        state['sorted_segments_list'].append(sorted_segments)
        if verbose:
            state['progress_bar'].update(1)

    def on_test_end(state):
        if verbose:
            state['progress_bar'].close()
            print()
        data = test_dataset.data
        eval_predictions(state['sorted_segments_list'], data)
        save_scores(state['scores'], data, dataset_name, split)

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, dataloader)

if __name__ == '__main__':
    split = args.split
    test_model(split)