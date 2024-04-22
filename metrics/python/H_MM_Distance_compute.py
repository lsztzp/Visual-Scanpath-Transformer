import os
import numpy as np
import torch

''' 该评价标准 分数越低 结果越好 '''
''' 
返回两个长度为4的分数集合（k=2,3,4,5），下标与对应k值差2，
如 H_scores[0] 代表 k=2时，预测序列中所有长度为2的子片段与其对应真值最优匹配的距离最大值， MM_scores则代表相应均值。
'''


def func_tde_trans(position, k):
    num_set = len(position) - k + 1
    mat = np.zeros([num_set, k])
    for i in range(k):
        mat[:, i] = position[i:num_set + i]
    return mat


def func_tde_metric(x_mat_pre, y_mat_pre, x_mat_gt, y_mat_gt, k):
    num_p = x_mat_pre.shape[0]
    num_g = x_mat_gt.shape[0]
    score_mat = np.zeros(num_p)
    for i in range(num_p):
        mat = np.sum(np.sqrt(np.square(x_mat_gt - np.tile(x_mat_pre[i], (num_g, 1))) + np.square(
            y_mat_gt - np.tile(y_mat_pre[i], (num_g, 1)))), 1)
        min_val = np.min(mat)
        score_mat[i] = min_val / k
    # print score_mat
    score_H = np.max(score_mat)
    score_M = np.mean(score_mat)
    return score_H, score_M


def H_MM_Distance(pre, gt):
    H_scores = np.ones(4) * 1000
    MM_scores = np.ones(4) * 1000
    for num_k in range(2, 6):
        x_position_ori = pre[:, 1]
        y_position_ori = pre[:, 0]
        if len(x_position_ori) < num_k:
            continue
        x_mat_pre = func_tde_trans(x_position_ori, num_k)
        y_mat_pre = func_tde_trans(y_position_ori, num_k)

        x_position_ori_1 = gt[:, 1]
        y_position_ori_1 = gt[:, 0]
        if len(x_position_ori_1) < num_k:
            continue
        x_mat_gt = func_tde_trans(x_position_ori_1, num_k)
        y_mat_gt = func_tde_trans(y_position_ori_1, num_k)

        score_H_pre, score_M_pre = func_tde_metric(x_mat_pre, y_mat_pre, x_mat_gt, y_mat_gt, num_k)
        H_scores[num_k - 2] = score_H_pre
        MM_scores[num_k - 2] = score_M_pre
    return H_scores, MM_scores


def H_MM_Distance_all(pre, gt_all, index):
    H_scores = np.ones(4) * 1000
    MM_scores = np.ones(4) * 1000
    for num_k in range(2, 6):
        x_position_ori = pre[:, 0]
        y_position_ori = pre[:, 1]
        if len(x_position_ori) < num_k:
            continue
        x_mat_pre = func_tde_trans(x_position_ori, num_k)
        y_mat_pre = func_tde_trans(y_position_ori, num_k)

        x_mat_gt_all = []
        y_mat_gt_all = []
        num_gts = len(gt_all)
        for i in range(num_gts):
            if i == index:
                continue
            gt = gt_all[i][1]
            x_position_ori_1 = gt[:, 1]
            y_position_ori_1 = gt[:, 0]
            if len(x_position_ori_1) < num_k:
                continue
            x_mat_gt = func_tde_trans(x_position_ori_1, num_k)
            y_mat_gt = func_tde_trans(y_position_ori_1, num_k)
            if len(x_mat_gt_all) == 0:
                x_mat_gt_all = x_mat_gt
            else:
                x_mat_gt_all = np.vstack((x_mat_gt_all, x_mat_gt))
            if len(y_mat_gt_all) == 0:
                y_mat_gt_all = y_mat_gt
            else:
                y_mat_gt_all = np.vstack((y_mat_gt_all, y_mat_gt))
        score_H_pre, score_M_pre = func_tde_metric(x_mat_pre, y_mat_pre, x_mat_gt_all, y_mat_gt_all, num_k);
        H_scores[num_k - 2] = score_H_pre
        MM_scores[num_k - 2] = score_M_pre
    return H_scores, MM_scores






