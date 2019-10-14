import argparse
import json
import pickle as pkl
import numpy as np

from eval import eval_predictions

possible_segments = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                     (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                     (0, 2), (1, 3), (2, 4), (3, 5),
                     (0, 3), (1, 4), (2, 5),
                     (0, 4), (1, 5),
                     (0, 5)]


def late_fusion(rgb_rgb_tag, rgb_flow_tag, flow_rgb_tag, flow_flow_tag, split, dataset_name, lambda_interval):
    if dataset_name == 'DiDeMo':
        data = json.load(open('/localdisk/szhang83/Developer/LocalizingMoments/data/%s_data.json' % split, 'r'))
    elif dataset_name == 'TEMPO_HL':
        data = json.load(open('/localdisk/szhang83/Developer/LocalizingMoments/data/tempoHL+didemo_%s.json' % split, 'r'))
    elif dataset_name == 'TEMPO_TL':
        data = json.load(open('/localdisk/szhang83/Developer/LocalizingMoments/data/tempoTL+didemo_%s.json' % split, 'r'))

    rgb_rgb_results = pkl.load(open(rgb_rgb_tag, 'rb'))
    rgb_flow_results = pkl.load(open(rgb_flow_tag, 'rb'))
    flow_rgb_results = pkl.load(open(flow_rgb_tag, 'rb'))
    flow_flow_results = pkl.load(open(flow_flow_tag, 'rb'))

    # data = list(filter(lambda d: d['annotation_id'].split('_')[0] not in ['while', 'before', 'after', 'then'], data))
    # data = list(filter(lambda d: d['annotation_id'].split('_')[0] == 'before', data))
    # data = list(filter(lambda d: d['annotation_id'].split('_')[0] == 'after', data))
    # data = list(filter(lambda d: d['annotation_id'].split('_')[0] == 'then', data))
    # data = list(filter(lambda d: d['annotation_id'].split('_')[0] == 'while', data))
    # # #
    best_rank1, best_rank5, best_miou = 0, 0, 0
    for i in np.arange(lambda_interval,1+lambda_interval,lambda_interval):
        for j in np.arange(lambda_interval,1+lambda_interval-i,lambda_interval):
            for k in np.arange(lambda_interval, 1 + lambda_interval-j-i, lambda_interval):
                l = 1-i-j-k
                all_segments = []
                for d in data:
                    rgb_rgb_scores = rgb_rgb_results[str(d['annotation_id'])]
                    rgb_flow_scores = rgb_flow_results[str(d['annotation_id'])]
                    flow_rgb_scores = flow_rgb_results[str(d['annotation_id'])]
                    flow_flow_scores = flow_flow_results[str(d['annotation_id'])]
                    scores = i*rgb_rgb_scores + j*rgb_flow_scores + k*flow_rgb_scores + l*flow_flow_scores
                    sorted_index = np.argsort(scores)[::-1]
                    all_segments.append([possible_segments[i] for i in sorted_index])
                rank1, rank5, miou = eval_predictions(all_segments, data,verbose=False)
                indicator = 0
                if rank1 > best_rank1: indicator += 1
                if rank5 > best_rank5: indicator += 1
                if miou > best_miou: indicator += 1
                if indicator > 1:
                # if rank1 > best_rank1:
                    best_rank1, best_rank5, best_miou = rank1, rank5, miou
                    print("i: %.1f, j: %.1f, k: %.1f, l: %.1f" % (i, j, k, l))
                    print("Average rank@1: %f" % rank1)
                    print("Average rank@5: %f" % rank5)
                    print("Average iou: %f" % miou)

    for i in np.arange(0,1+lambda_interval,lambda_interval):
        j,k = 0,0
        l = 1-i
        all_segments = []
        for d in data:
            rgb_rgb_scores = rgb_rgb_results[str(d['annotation_id'])]
            rgb_flow_scores = rgb_flow_results[str(d['annotation_id'])]
            flow_rgb_scores = flow_rgb_results[str(d['annotation_id'])]
            flow_flow_scores = flow_flow_results[str(d['annotation_id'])]
            scores = i*rgb_rgb_scores + j*rgb_flow_scores + k*flow_rgb_scores + l*flow_flow_scores
            sorted_index = np.argsort(scores)[::-1]
            all_segments.append([possible_segments[i] for i in sorted_index])
        rank1, rank5, miou = eval_predictions(all_segments, data,verbose=False)
        print("i: %.1f, j: %.1f, k: %.1f, l: %.1f" % (i, j, k, l))
        print("Average rank@1: %f" % rank1)
        print("Average rank@5: %f" % rank5)
        print("Average iou: %f" % miou)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    from config import path_prefix

    parser.add_argument("--rgb_rgb_tag", type=str, default=path_prefix+'results/TEMPO_TL_test_rgb-rgb.pkl')
    parser.add_argument("--rgb_flow_tag", type=str, default=path_prefix+'results/TEMPO_TL_test_rgb-flow.pkl')
    parser.add_argument("--flow_rgb_tag", type=str, default=path_prefix+'results/TEMPO_TL_test_flow-rgb.pkl')
    parser.add_argument("--flow_flow_tag", type=str, default=path_prefix+'results/TEMPO_TL_test_flow-flow.pkl')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--dataset_name", type=str, default='TEMPO_TL')
    parser.add_argument('--lambda_interval', type=float, default=0.1)

    args = parser.parse_args()

    late_fusion(args.rgb_rgb_tag, args.rgb_flow_tag, args.flow_rgb_tag, args.flow_flow_tag,
                args.split, args.dataset_name, args.lambda_interval)