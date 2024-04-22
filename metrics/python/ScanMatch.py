import numpy as np


def ScanMatch(test_seq, gt_seq, ScanMatchInfo_):
    if len(test_seq) == 0 or len(gt_seq) == 0:
        score = 0
    else:
        test_num = ScanMatch_FixationToSequence(test_seq, ScanMatchInfo_)
        gt_num = ScanMatch_FixationToSequence(gt_seq, ScanMatchInfo_)
        score = ScanMatch_compute(test_num, gt_num, ScanMatchInfo_)
    return score


def ScanMatch_FixationToSequence(seq, ScanMatchInfo_):
    mask = ScanMatchInfo_['mask'][0][0]
    num_fixations = len(seq)
    seq_num = np.zeros(num_fixations)
    for i in range(num_fixations):
        seq_num[i] = mask[int(seq[i][0]), int(seq[i][1])]
    seq_num -= 1
    return seq_num


def ScanMatch_compute(seq1, seq2, ScanMatchInfo_):
    m = len(seq1)
    n = len(seq2)

    ScoringMatrix = ScanMatchInfo_['SubMatrix'][0][0]

    gap = ScanMatchInfo_['GapValue']

    gap = gap[0][0]
    best_matrix = np.zeros([n + 1, m + 1])
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                best_matrix[i][j] = gap * j
            elif j == 0:
                best_matrix[i][j] = gap * i
            else:
                match = ScoringMatrix[int(seq2[i - 1]), int(seq1[j - 1])]
                gap1_score = best_matrix[i - 1][j] + gap
                gap2_score = best_matrix[i][j - 1] + gap
                match_score = best_matrix[i - 1][j - 1] + match
                best_matrix[i][j] = max(gap1_score, gap2_score, match_score)
    score = np.max(best_matrix)
    max_sub = np.max(ScoringMatrix)
    scale = max_sub * max(m, n)
    score = float(score) / scale

    return score
