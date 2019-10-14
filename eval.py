'''
Code to evaluate your results on the DiDeMo dataset.
'''

import numpy as np
import json


def iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection) / union


def rank(pred, gt):
    return pred.index(tuple(gt)) + 1


def eval_predictions(segments, data, verbose=True, is_context=False):
    '''
    Inputs:
	segments: For each item in the ground truth data, rank possible video segments given the description and video.  In DiDeMo, there are 21 posible moments extracted for each video so the list of video segments will be of length 21.  The first video segment should be the video segment that best corresponds to the text query.  There are 4180 sentence in the validation data, so when evaluating a model on the val dataset, segments should be a list of lenght 4180, and each item in segments should be a list of length 21.
	data: ground truth data
    '''
    rank1 = []
    rank5 = []
    miou = []

    key = 'context' if is_context else 'times'

    # DiDeMo
    average_ranks = []
    average_iou = []
    for s, d in zip(segments, data):
        if isinstance(d['annotation_id'], int):
            d['annotation_id'] = str(d['annotation_id'])
        if d['annotation_id'].split('_')[0] in ['before', 'after', 'then', 'while']:
            continue
        pred = s[0]
        d[key] = [d[key] if len(d[key])>0 else [0,d['num_segments']-1]] if is_context else d[key]
        ious = [iou(pred, t) for t in d[key]]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [rank(s, t) for t in d[key]]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    if len(average_ranks) > 0:
        rank1.append(np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks)))
        rank5.append(np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks)))
        miou.append(np.mean(average_iou))

    # Before
    average_ranks = []
    average_iou = []
    for s, d in zip(segments, data):
        if isinstance(d['annotation_id'], int):
            d['annotation_id'] = str(d['annotation_id'])
        if d['annotation_id'].split('_')[0] != 'before':
            continue
        pred = s[0]
        d[key] = [d[key] if len(d[key])>0 else [0,d['num_segments']-1]] if is_context else d[key]
        ious = [iou(pred, t) for t in d[key]]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [rank(s, t) for t in d[key]]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    if len(average_ranks) > 0:
        rank1.append(np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks)))
        rank5.append(np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks)))
        miou.append(np.mean(average_iou))

    # After
    average_ranks = []
    average_iou = []
    for s, d in zip(segments, data):
        if isinstance(d['annotation_id'], int):
            d['annotation_id'] = str(d['annotation_id'])
        if d['annotation_id'].split('_')[0] != 'after':
            continue
        pred = s[0]
        d[key] = [d[key] if len(d[key])>0 else [0,d['num_segments']-1]] if is_context else d[key]
        ious = [iou(pred, t) for t in d[key]]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [rank(s, t) for t in d[key]]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    if len(average_ranks) > 0:
        rank1.append(np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks)))
        rank5.append(np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks)))
        miou.append(np.mean(average_iou))

    # Then
    average_ranks = []
    average_iou = []
    for s, d in zip(segments, data):
        if isinstance(d['annotation_id'], int):
            d['annotation_id'] = str(d['annotation_id'])
        if d['annotation_id'].split('_')[0] != 'then':
            continue
        pred = s[0]
        d[key] = [d[key] if len(d[key])>0 else [0,d['num_segments']-1]] if is_context else d[key]
        ious = [iou(pred, t) for t in d[key]]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [rank(s, t) for t in d[key]]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    if len(average_ranks) > 0:
        rank1.append(np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks)))
        rank5.append(np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks)))
        miou.append(np.mean(average_iou))

    # While
    average_ranks = []
    average_iou = []
    for s, d in zip(segments, data):
        if isinstance(d['annotation_id'], int):
            d['annotation_id'] = str(d['annotation_id'])
        if d['annotation_id'].split('_')[0] != 'while':
            continue
        pred = s[0]
        d[key] = [d[key] if len(d[key])>0 else [0,d['num_segments']-1]] if is_context else d[key]
        ious = [iou(pred, t) for t in d[key]]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [rank(s, t) for t in d[key]]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    if len(average_ranks) > 0:
        rank1.append(np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks)))
        rank5.append(np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks)))
        miou.append(np.mean(average_iou))


    rank1 = np.mean(rank1)
    rank5 = np.mean(rank5)
    miou = np.mean(miou)

    if verbose:
        print("Average rank@1: %f" % rank1)
        print("Average rank@5: %f" % rank5)
        print("Average iou: %f" % miou)
    return rank1, rank5, miou

if __name__ == '__main__':

    '''
    Example code to evaluate your model.  Below I compute the scores for the moment frequency prior
    You should see the following output when you run eval.py
        Average rank@1: 0.186842
        Average rank@5: 0.686842
    	Average iou: 0.252216

        HL Context with global:
        Average rank@1: 0.376474
        Average rank@5: 0.784819
    	Average iou: 0.521964

        TL Context with global:
        Average rank@1: 0.173985
        Average rank@5: 0.690257
    	Average iou: 0.369661

        HL Context without global:
        Average rank@1: 0.250000
        Average rank@5: 0.731024
    	Average iou: 0.407356

        TL Context without global:
        Average rank@1: 0.000000
        Average rank@5: 0.635300
    	Average iou: 0.220570
    '''

    train_data = json.load(open('/localdisk/szhang83/Developer/LocalizingMoments/data/tempoTL+didemo_train.json', 'r'))
    val_data = json.load(open('/localdisk/szhang83/Developer/LocalizingMoments/data/tempoTL+didemo_test.json', 'r'))
    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in [d['context'] if len(d['context'])>0 else [0,d['num_segments']-1]]]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prediction = [prior for d in val_data]

    eval_predictions(prediction, val_data, is_context=True)