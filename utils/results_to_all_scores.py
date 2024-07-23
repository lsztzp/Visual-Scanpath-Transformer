import os
from metrics.utils import get_score_file
# from metrics.matlab.matlab_score_gen import get_matlab_scores


def resultsToScores(work_dir, epoch=0):
    dataset_names = ('OSIE', 'MIT', 'SALICON', 'iSUN')
    with open(os.path.join(work_dir, 'scores.txt'), 'w') as f:  # file object
        if epoch:
            f.write(f"results------------------------{epoch}-----------------:\n")  # to write
        for dataset_name in dataset_names:
            scores_all_data = get_score_file(os.path.join(work_dir, dataset_name), dataset_name,
                                             metrics=('scanmatch', 'tde', 'mutimatch', 'dtw',))

            f.write(f"{dataset_name}---------python--scores---------:\n")
            for key, value in scores_all_data.items():
                f.write(f"{key}:   {value}\n")

            for key, value in scores_all_data.items():
                print(f"{key}:   {value}\n")


            # matlab_scores = get_matlab_scores(data_path=os.path.join(work_dir, dataset_name) + '/',
            #                                   dataset_name=dataset_name,
            #                                   metrics=('scanmatch', 'ss_w', 'tde',))

            # f.write(f"{dataset_name}---------matlab--scores----------:\n")  # to write
            # for key, value in matlab_scores.items():
            #     f.write(f"{key}:   {value}\n")
            f.write(f"########################################################################\n\n")

            for key, value in scores_all_data.items():
                print(f"{key}:   {value}\n")

        # for key, value in scores_all_data.items():
        #     print(f"{key}:   {value:}\n")
        #
        # for key, value in matlab_scores.items():
        #     print(f"{key}:   {value:}\n")