import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from data_processing import possible_segments
import dataset
from model.model import MLLC_FullAtt
from engine import Engine
from metrics import AverageMeter
import eval
from config import args, device, path_prefix


torch.manual_seed(0)
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    dataset_name = args.dataset_name

    input_size_0 = 4096 if args.feature_type_0 == 'rgb' else 1024
    input_size_1 = 4096 if args.feature_type_1 == 'rgb' else 1024

    model = MLLC_FullAtt(input_size_0=input_size_0, input_size_1=input_size_1, txt_input_size=args.lang_hidden_size, hidden_size=args.hidden_size).to(device)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-8)#weight decay to 5e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=20, verbose=True)

    train_dataset = getattr(dataset, dataset_name)('train')
    val_dataset = getattr(dataset, dataset_name)('val')
    test_dataset = getattr(dataset, dataset_name)('test')

    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    collate_fn=train_dataset.collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    collate_fn=val_dataset.collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    collate_fn=test_dataset.collate_fn)
        else:
            raise ValueError

        return dataloader

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

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/args.batch_size*args.test_interval)
        state['t'] = 1
        model.train()
        if args.verbose:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if args.verbose:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if args.verbose:
                state['progress_bar'].close()

            val_state = engine.test(network, iterator('val'))
            test_state = engine.test(network, iterator('test'))
            state['scheduler'].step(-val_state['loss_meter'].avg)

            saved_model_filename = path_prefix + 'checkpoint/{}/{}-{}-H{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                dataset_name, args.feature_type_0, args.feature_type_1, args.hidden_size,
                state['t'], val_state['rank1'], val_state['miou'])

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)
            torch.save(model.state_dict(), saved_model_filename)

            print('iter: {} train loss {:.4f} '
                  'val loss {:.4f} rank@1: {:.4f} rank@5: {:.4f} miou: {:.4f} '
                  'test loss {:.4f} rank@1: {:.4f} rank@5: {:.4f} miou: {:.4f}'.format(
                state['t'], state['loss_meter'].avg,
                val_state['loss_meter'].avg, val_state['rank1'], val_state['rank5'], val_state['miou'],
                test_state['loss_meter'].avg, test_state['rank1'], test_state['rank5'], test_state['miou'],))
            sys.stdout.flush()
            if args.verbose:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            val_state['loss_meter'].reset()
            test_state['loss_meter'].reset()
            state['loss_meter'].reset()

    def on_end(state):
        if args.verbose:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []

    def on_test_forward(state):
        state['loss_meter'].update(state['loss'].item(), 1)

        scores = state['output'].cpu().data.numpy().squeeze()
        sorted_index = np.argsort(scores)[::-1]
        sorted_segments = [possible_segments[i] for i in sorted_index]
        state['sorted_segments_list'].append(sorted_segments)

    def on_test_end(state):
        data = state['iterator'].dataset.data
        state['rank1'], state['rank5'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], data, verbose=False)

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=args.max_epoch,
                 optimizer=optimizer,
                 scheduler=scheduler)
