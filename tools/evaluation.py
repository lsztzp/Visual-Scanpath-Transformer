import time
from scipy import io
from torch import nn
from modules.mdn import mixture_probability
from dataset.dataloder import ScanpathDataloder
from metrics.utils import get_score_filename
from utils.utils import *
import torch
from collections import Iterable
criterion = nn.CrossEntropyLoss()


class Evaluation:
    def __init__(self, feature_extrator, work_dir, writer, val_batch_size, device, seed, max_length,
                 action_map_size, input_resize, patch_size,):

        super(Evaluation, self).__init__()
        self.feature_extrator = feature_extrator
        self.work_dir = work_dir
        self.writer = writer
        self.val_batch_size = val_batch_size
        self.device = device
        self.max_length = max_length
        self.action_map_size = action_map_size

        self.osie_val_dataiter = ScanpathDataloder(dataset_name='osie', phase='test',
                                          batch_size=val_batch_size, input_resize=input_resize,
                                          patch_size=patch_size, max_length=max_length, seed=seed)
        self.salicon_val_dataiter = ScanpathDataloder(dataset_name='salicon', phase='val',
                                             batch_size=val_batch_size, input_resize=input_resize,
                                          patch_size=patch_size, max_length=max_length,seed=seed)
        self.mit_val_dataiter = ScanpathDataloder(dataset_name='mit', phase='test',
                                         batch_size=val_batch_size, input_resize=input_resize,
                                          patch_size=patch_size, max_length=max_length,seed=seed)
        self.isun_val_dataiter = ScanpathDataloder(dataset_name='isun', phase='test',
                                          batch_size=val_batch_size, input_resize=input_resize,
                                          patch_size=patch_size, max_length=max_length,seed=seed)

    def validation(self, model, epoch, dataset_name, save=False,):

        model.eval()
        if dataset_name == 'isun':
            save_dir_path = os.path.join(self.work_dir, "seq_results_best_model", "iSUN")
        else:
            save_dir_path = os.path.join(self.work_dir, "seq_results_best_model", dataset_name.upper())
        if save and not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        with torch.no_grad():
            val_performance, seq_count = {'loss': 0, 'len_error': 0}, 0
            if dataset_name == 'salicon':
                val_data_iter = self.salicon_val_dataiter
            elif dataset_name == 'osie':
                val_data_iter = self.osie_val_dataiter
            elif dataset_name == 'mit':
                val_data_iter = self.mit_val_dataiter
            elif dataset_name == 'isun':
                val_data_iter = self.isun_val_dataiter

            for i_batch, batch in enumerate(val_data_iter):
                val_batch_size = batch['sals'].size(0)

                hw_t = get_hw_t(self.action_map_size[0], self.action_map_size[1]).to(self.device)
                enc_masks = None
                dec_inputs = torch.ones(val_batch_size, 1, 2).to(self.device) * 0.5
                imgs = batch['imgs'].to(self.device)
                sals = batch['sals'].to(self.device)

                start = time.time()
                enc_inputs = self.feature_extrator(imgs, sals)

                enc_outputs, enc_self_attns = model.encoder(enc_inputs, enc_masks,)
                for n in range(self.max_length):
                    dec_outputs, dec_self_attns, dec_enc_attns = model.decoder(enc_outputs, dec_inputs,
                                                                               enc_masks=enc_masks,
                                                                               dec_masks=torch.zeros(val_batch_size,
                                                                                                     n + 1).to(self.device))

                    pis, mus, sigmas, rhos = model.mdn(dec_outputs)

                    pred_roi_maps = model.mdn.mixture_probability_map(pis, mus, sigmas, rhos, hw_t).reshape(
                        (-1, n + 1, self.action_map_size[0], self.action_map_size[1])).flatten(2)

                    _, indexs = pred_roi_maps.max(-1)
                    indexs_w = ((indexs % self.action_map_size[1]) / self.action_map_size[1]).unsqueeze(-1)
                    indexs_h = (torch.div(indexs, self.action_map_size[1], rounding_mode='trunc') / self.action_map_size[0]).unsqueeze(-1)
                    outputs = torch.cat((indexs_h, indexs_w), -1)
                    if n == 0:
                        end1 = time.time()
                        print(end1 - start)


                    outputs = outputs.clamp(0, 0.99)
                    # dec_inputs = outputs
                    last_fixations = outputs[:, -1]
                    dec_inputs = torch.cat((dec_inputs, last_fixations.unsqueeze(1)), dim=1)

                len_error = abs(9 - batch['avg_lens']).mean().item()

                probs = mixture_probability(pis, mus, sigmas, rhos, batch['gt_fixations'][:, 0].unsqueeze(-1).to(self.device)).squeeze()

                probs_mask = torch.arange(self.max_length).expand(val_batch_size, self.max_length). \
                    lt(batch['valid_lens'][:, 0].unsqueeze(-1).expand(val_batch_size, self.max_length)).to(self.device)
                probs = torch.masked_select(probs, probs_mask)
                loss_fixation = torch.mean(-torch.log(probs))

                loss = loss_fixation
                loss_item = loss.detach().cpu().item()

                pred_roi_maps = model.mdn.mixture_probability_map(pis, mus, sigmas, rhos, hw_t).reshape(
                    (-1, self.max_length, self.action_map_size[0], self.action_map_size[1]))

                _, indexs = pred_roi_maps.flatten(2).max(-1)
                indexs_w = ((indexs % self.action_map_size[1]) / self.action_map_size[1]).unsqueeze(-1)
                indexs_h = (torch.div(indexs, self.action_map_size[1], rounding_mode='trunc') / self.action_map_size[0]).unsqueeze(-1)
                outputs = torch.cat((indexs_h, indexs_w), -1).clamp(0, 0.99).detach().cpu()

                outputs = outputs.clamp(min=0).numpy()

                if dataset_name == 'osie' or dataset_name == 'salicon':
                    clamp_len = 9
                elif dataset_name == 'mit':
                    clamp_len = 8
                elif dataset_name == 'isun':
                    clamp_len = 6

                for batch_index in range(len(batch['imgs'])):
                    output = outputs[batch_index][:clamp_len]
                    output[:, 0] *= batch['img_sizes'][batch_index][0].item()
                    output[:, 1] *= batch['img_sizes'][batch_index][1].item()

                    if save:
                        save_path = os.path.join(save_dir_path, batch['file_names'][batch_index]+'.mat')
                        io.savemat(save_path,
                                   {'fixations': output})

                    scores = get_score_filename(output, batch['file_names'][batch_index], dataset_name=dataset_name,
                                           metrics=('scanmatch', 'tde', 'mutimatch'))

                    val_performance['loss'] += loss_item
                    val_performance['len_error'] += len_error
                    for metric, score in scores.items():
                        if metric in val_performance:
                            val_performance[metric] += score
                        else:
                            val_performance[metric] = score

                    seq_count += 1
                    print('Epoch:', '%03d' % epoch, f"({seq_count}):", 'val_loss =', f'{loss_item:.6f}'
                                                                  f' val_score {scores}')

                if not save and seq_count > 700:
                    break

            if seq_count > 0:
                for metric, _ in scores.items():
                    val_performance[metric] /= seq_count

            print('epoch', epoch, f'{dataset_name}_val_performance', val_performance,)
            if not save:
                for key, value in val_performance.items():
                    print(key, value, isinstance(value, Iterable))
                    if isinstance(value, Iterable):
                        for index, sub_value in enumerate(value):
                            self.writer.add_scalar(f'val_{dataset_name}_{key}/score{index}', sub_value, epoch)
                    else:
                        self.writer.add_scalar(f'AA_Scalar/val_{dataset_name}_{key}', value, epoch)
        return val_performance['scanmatch']