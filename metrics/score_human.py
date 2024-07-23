import os
import numpy as np
from PIL import Image
from scipy import io
from metrics.utils import salicon_gtspath, osie_gtspath, score_all_gts, mit_gtspath, mit_imgspath, isun_gtspath, \
    isun_imgspath

def score_human(source, metrics, ):
    source = source.lower()
    print(f'comptering {source} dataset in {metrics}')
    imagespath = ""
    gtspath = ''
    if source == 'salicon':
        gtspath = salicon_gtspath
    elif source == 'osie':
        gtspath = osie_gtspath
    elif source == 'mit':
        gtspath = mit_gtspath
        imagespath = mit_imgspath
    elif source == 'isun':
        gtspath = isun_gtspath
        imagespath = isun_imgspath

    gtspathdir = os.listdir(gtspath)

    scores_all_data = {}
    count_data = 0
    for index in range(len(gtspathdir)):

        gt_name = gtspathdir[index]
        gt_path = os.path.join(gtspath, gt_name)

        gt_fixations = io.loadmat(gt_path)
        gt_fixations = gt_fixations['gt_fixations'][0]

        print(index, gt_fixations.shape)
        # reshape gtFixations
        if source == 'mit' or source == 'isun':
            image_path = os.path.join(imagespath, gt_name[:-4] + '.jpg')
            image = Image.open(image_path)
            h, w = image.height, image.width

            for n in range(len(gt_fixations)):
                gt_fixation = gt_fixations[n].astype(np.float)
                gt_fixation[:, 0] /= h
                gt_fixation[:, 1] /= w
                # gt_fixation = np.clip(gt_fixation, 0, 0.98)
                gt_fixation[:, 0] *= 600
                gt_fixation[:, 1] *= 800
                gt_fixations[n] = gt_fixation

        scores_all_human = {}
        count_human = 0
        for n in range(len(gt_fixations)):
            gt_fixation = gt_fixations[n].astype(np.float)
            other = np.delete(gt_fixations, n, axis=0)
            scores_gts = score_all_gts(gt_fixation, other, source, metrics)

            # add this human score
            for metric, score in scores_gts.items():
                if metric in scores_all_human:
                    scores_all_human[metric] += score
                else:
                    scores_all_human[metric] = score
            count_human += 1
        # process all human scores as this data score
        for metric, score in scores_all_human.items():
            scores_all_human[metric] /= count_human

        # add this data score
        for metric, score in scores_all_human.items():
            if metric in scores_all_data:
                scores_all_data[metric] += score
            else:
                scores_all_data[metric] = score
        count_data += 1

    # prcess all data scores
    for metric, score in scores_all_data.items():
        scores_all_data[metric] /= count_data

    return scores_all_data

sources = ('OSIE', 'MIT', 'SALICON', 'iSUN')
# metrics = ('scanmatch', 'tde', 'mutimatch')
metrics = ('dtw', )

results = []
for source in sources:
    results.append(score_human(source, metrics))
print(results)


