import numpy as np
import os
import multimatch_gaze as m
import scipy.io as scio
from PIL import Image
from dtaidistance import dtw_ndim
from metrics.python.H_MM_Distance_compute import H_MM_Distance
from metrics.python.ScanMatch import ScanMatch
from metrics.python.scanmatch_vqa import ScanMatchwithDuration, ScanMatchwithoutDuration

ScanMatchInfo_salicon = scio.loadmat('metrics/python/SALICON_ScanMatchInfo.mat')['ScanMatchInfo']
ScanMatchInfo_osie = scio.loadmat('metrics/python/OSIE_ScanMatchInfo.mat')['ScanMatchInfo']

salicon_gtspath = '/data/qmengyu/01-Datasets/01-ScanPath-Dataset/SALICON/gt_fixations_test/'
osie_gtspath = '/data/qmengyu/01-Datasets/01-ScanPath-Dataset/OSIE/gt_fixations_test/'

mit_gtspath = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/MIT/gt_fixations/"
mit_imgspath = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/MIT/images/all/"

isun_gtspath = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/iSUN/gt_fixations/"
isun_imgspath = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/iSUN/images/"


def score_seq(pre, gt, dataset_name, metrics=('scanmatch', 'tde', 'mutimatch', 'dtw')):
    if dataset_name == 'salicon':
        image_size = [480, 640]
        ScanMatchInfo = ScanMatchInfo_salicon
    else:
        image_size = [600, 800]
        ScanMatchInfo = ScanMatchInfo_osie

    scores = {}
    if 'scanmatch' in metrics:
        scores['scanmatch'] = ScanMatch(pre.astype(np.int), gt.astype(np.int), ScanMatchInfo)
    if 'tde' in metrics:
        tde_h, tde_m = H_MM_Distance(pre.astype(np.int), gt.astype(np.int))
        scores['tde_h'], scores['tde_m'] = np.array(tde_h), np.array(tde_m)

    if 'dtw' in metrics:
        scores['dtw'] = dtw_ndim.distance(pre.astype(np.int), gt.astype(np.int))

    if 'mutimatch' in metrics:
        pre = np.array([(loc[0], loc[1], 0.1) for loc in pre],
                       [('start_x', float), ('start_y', float), ('duration', float)]).view(np.recarray)

        gt = np.array([(loc[0], loc[1], 0.1) for loc in gt],
                      [('start_x', float), ('start_y', float), ('duration', float)]).view(np.recarray)

        scores['mutimatch'] = np.array(m.docomparison(pre, gt, screensize=[image_size[0], image_size[1]]))[:-1]

    return scores

def score_all_gts(pred_fixation, gt_fixations, dataset_name, metrics=('scanmatch', 'tde', 'mutimatch', 'dtw',)):

    scores_all_gts = {}
    count = 0
    for n in range(len(gt_fixations)):
        gt_fixation = gt_fixations[n].astype(np.float)
        if len(gt_fixation) >= 3 and len(pred_fixation) >= 3:

            scores = score_seq(pred_fixation.astype(np.int), gt_fixation.astype(np.int), dataset_name, metrics)

            for metric, score in scores.items():
                if metric in scores_all_gts:
                    scores_all_gts[metric] += score
                else:
                    scores_all_gts[metric] = score
            count += 1
    if count != 0:
        for metric, score in scores_all_gts.items():
            scores_all_gts[metric] /= count

    return scores_all_gts


def get_score_filename(pred_fixation, file_name, dataset_name, metrics=('scanmatch', 'tde', 'mutimatch', 'dtw',)):
    imagespath = ""
    gtspath = ''
    if dataset_name == 'salicon':
        gtspath = salicon_gtspath
    elif dataset_name == 'osie' or dataset_name == 'osie1':
        gtspath = osie_gtspath
    elif dataset_name == 'mit':
        gtspath = mit_gtspath
        imagespath = mit_imgspath
    elif dataset_name == 'isun':
        gtspath = isun_gtspath
        imagespath = isun_imgspath

    gt_path = os.path.join(gtspath, file_name + '.mat')

    gt_fixations = scio.loadmat(gt_path)
    gt_fixations = gt_fixations['gt_fixations'][0]

    if dataset_name == 'osie':
        pred_fixation[:, 0] /= 600
        pred_fixation[:, 1] /= 800

        pred_fixation = np.clip(pred_fixation, 0, 0.98)

        pred_fixation[:, 0] *= 600
        pred_fixation[:, 1] *= 800

    if dataset_name == 'salicon':
        pred_fixation[:, 0] /= 480
        pred_fixation[:, 1] /= 640

        pred_fixation = np.clip(pred_fixation, 0, 0.98)

        pred_fixation[:, 0] *= 480
        pred_fixation[:, 1] *= 640

    if dataset_name == 'mit' or dataset_name == 'isun':
        image_path = os.path.join(imagespath, file_name + '.jpg')
        image = Image.open(image_path)
        h, w = image.height, image.width

        pred_fixation[:, 0] /= h
        pred_fixation[:, 1] /= w

        pred_fixation = np.clip(pred_fixation, 0, 0.98)

        pred_fixation[:, 0] *= 600
        pred_fixation[:, 1] *= 800
        for n in range(len(gt_fixations)):
            gt_fixation = gt_fixations[n].astype(np.float)
            gt_fixation[:, 0] /= h
            gt_fixation[:, 1] /= w
            gt_fixation = np.clip(gt_fixation, 0, 0.98)
            gt_fixation[:, 0] *= 600
            gt_fixation[:, 1] *= 800
            gt_fixations[n] = gt_fixation

    return score_all_gts(pred_fixation, gt_fixations, dataset_name, metrics)


def get_score_file(pred_file_path, dataset_name, metrics=('scanmatch', 'tde', 'mutimatch', 'dtw',)):
    dataset_name = dataset_name.lower()
    scores_all_data = {}
    count_data = 0
    predspathdir = os.listdir(pred_file_path)
    predspathdir.sort()
    for index in range(len(predspathdir)):
        print(index)
        file_name = predspathdir[index][:-4]

        pred_path = os.path.join(pred_file_path, file_name+'.mat')
        pred_fixations = scio.loadmat(pred_path)['fixations'].astype(float)
        # print(pred_fixations[:, 0].max())
        scores_all_gts = get_score_filename(pred_fixations, file_name, dataset_name, metrics=metrics)

        for metric, score in scores_all_gts.items():
            if metric in scores_all_data:
                scores_all_data[metric] += score
            else:
                scores_all_data[metric] = score
        count_data += 1

    for metric, score in scores_all_data.items():
        scores_all_data[metric] /= count_data
    return scores_all_data

