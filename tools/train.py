from modules.mdn import mixture_probability
from utils.utils import *


class Trainer:
    def __init__(self, lr, dataloder, work_dir, device, start_epoch, epoch_nums,
                train_gt_num, val_step,
                writer, eos=False, lambda_1=0):
        self.lr = lr
        self.dataloder = dataloder
        self.work_dir = work_dir
        self.device = device
        self.train_gt_num = train_gt_num
        self.cur_epoch = start_epoch
        self.epoch_nums = epoch_nums
        self.val_step = val_step
        self.writer = writer
        self.eos = eos
        self.lambda_l = lambda_1

    def train_epochs(self, feature_extrator, model, optimizer, lr_scheduler, evaluation,):
        if lr_scheduler:
            scheduler = build_scheduler(lr_scheduler=lr_scheduler, optimizer=optimizer)

        criterion = torch.nn.CrossEntropyLoss()

        best_epoch, best_score = 0, 0
        for self.cur_epoch in range(self.cur_epoch, self.epoch_nums + 1):
            model.train()
            train_performance, batch_count = {'loss': 0}, 0
            for i_batch, batch in enumerate(self.dataloder):
                optimizer.zero_grad()

                enc_inputs = feature_extrator(batch['imgs'].to(self.device), batch['sals'].to(self.device))
                enc_masks = None
                loss_fixations_all = list()
                train_batch_size = batch['imgs'].shape[0]
                max_length = batch['dec_inputs'].shape[-2]
                for seq_n in range(self.train_gt_num):
                    pis, mus, sigmas, rhos, eos = model(enc_inputs, enc_masks,
                                                        batch['dec_inputs'][:, seq_n].float().to(self.device),
                                                        batch['dec_masks'][:, seq_n].to(self.device))

                    probs = mixture_probability(pis, mus, sigmas, rhos,
                                                batch['gt_fixations'][:, seq_n].unsqueeze(-1).to(self.device)).squeeze()
                    probs_mask = torch.arange(max_length).expand(train_batch_size, max_length). \
                        lt(batch['valid_lens'][:, seq_n].unsqueeze(-1).expand(train_batch_size, max_length)).to(self.device)
                    probs = torch.masked_select(probs, probs_mask)
                    loss_fixation = torch.mean(-torch.log(probs))

                    loss_fixations_all.append(loss_fixation.unsqueeze(0))

                loss_fixations = torch.cat(loss_fixations_all, dim=0).mean()
                if self.eos:
                    # loss_len = torch.cat(loss_fixations_all, dim=0).mean()
                    eos_gt = torch.arange(max_length).expand(train_batch_size, max_length). \
                        ge((batch['avg_lens'] - 1).unsqueeze(-1).expand(train_batch_size, max_length)).flatten().to(self.device)
                    loss_len = criterion(self.eos.reshape(-1, 2), eos_gt.long())
                    loss = loss_fixations + self.lambda_l * loss_len
                else:
                    loss = loss_fixations

                batch_count += 1
                loss_item = loss.detach().cpu().item()
                train_performance['loss'] += loss_item

                print(f'train_(epoch{self.cur_epoch:3d}/{i_batch:4d}), ' + f'loss: {loss_item:.3f}')
                loss.backward()
                optimizer.step()

            train_performance['loss'] /= batch_count
            self.writer.add_scalar('AA_Scalar/train_loss', train_performance['loss'], self.cur_epoch)
            self.writer.add_scalar('AA_Scalar/train_lr', float(optimizer.param_groups[0]['lr']), self.cur_epoch)

            if lr_scheduler:
                scheduler.step()

            if self.cur_epoch % self.val_step == 0:
                save_checkpoint(self.cur_epoch, model, optimizer, self.work_dir)
                score = evaluation.validation(model, self.cur_epoch, dataset_name='osie', save=False)
                if score > best_score:
                    best_epoch = self.cur_epoch

        return best_epoch
