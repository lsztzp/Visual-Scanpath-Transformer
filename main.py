import argparse
from torch import optim
from dataset.dataloder import ScanpathDataloder
from tools.evaluation import Evaluation
from torch.utils.tensorboard import SummaryWriter
from modules.transformer import Transformer
from modules.vgg import VGG_Gen
from tools.train import Trainer
from utils.results_to_all_scores import resultsToScores
from utils.utils import *
from mmcv import Config, DictAction

from modules.salgan import Feature_Extrator
from datetime import datetime

parser = argparse.ArgumentParser(description='Train a model or Inference')
parser.add_argument('--config', default='config.py', help='config.py path')
parser.add_argument('--work_dir', default='Baseline', help='path to save logs and weights')
parser.add_argument('--device', default='cuda:0', help='cuda:n')
parser.add_argument('--wo_train', action="store_true", help='w/o train the model') # w/o means without
parser.add_argument('--wo_inference', action="store_true", help='w/o inference to scanpath results?')
parser.add_argument('--wo_score', action="store_true", help='w/o score scanpath results?')
parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

args = parser.parse_args()
cfg = Config.fromfile(args.config)
cfg.merge_from_dict(vars(args))

if args.options is not None:
    cfg.merge_from_dict(args.options)

if not cfg.wo_train:
    cfg.work_dir = os.path.join('./logs/', datetime.today().strftime('%m-%d-') + cfg.work_dir)
else:
    assert cfg.reload_path
    cfg.work_dir = os.path.join('./logs/', cfg.work_dir)

writer = SummaryWriter(log_dir=cfg.work_dir)
setup_seed(cfg.seed)

# select backbone
if cfg.vgg_fea:
    feature_extrator = VGG_Gen(num_patch_h=cfg.num_patch_h, num_patch_w=cfg.num_patch_w, device=cfg.device)
else:
    feature_extrator = Feature_Extrator(input_resize=cfg.input_resize, feature_dim=cfg.feature_dim,
                                    patch_size=cfg.patch_size, saliency_attention=cfg.saliency_attention, sal_gen=cfg.sal_gen)

feature_extrator = feature_extrator.to(cfg.device)
model = Transformer(cfg.feature_dim,
                    cfg.num_patch_h,
                    cfg.num_patch_w,
                    cfg.d_model,
                    cfg.d_k,
                    cfg.d_v,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                    cfg.enc_n_layers,
                    cfg.postion_method,
                    cfg.max_length,
                    cfg.dec_n_layers,
                    cfg.MDN_hidden_num,
                    cfg.num_gauss,
                    cfg.eos).to(cfg.device)

if cfg.backbone_grad:
    optimizer = optim.AdamW([{'params': model.parameters()},
                {'params': feature_extrator.parameters(), 'lr': cfg.backbone_lr}], lr=cfg.lr, weight_decay=cfg.weight_decay)
else:
    for p in feature_extrator.parameters():
        p.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

if cfg.reload_path: # finetune or score: load checkpiont from file reload_path
    epoch_start, model, optimizer = loadCheckpoint(model=model, optimizer=optimizer, checkpointPath=cfg.reload_path)
else: # resume from break-point: reload checkpoint from dir workdir
    epoch_start, model, optimizer = loadCheckpoint(model=model, optimizer=optimizer, work_dir=cfg.work_dir)

print('model training results will be in :',cfg.work_dir)

cfg.dump(os.path.join(cfg.work_dir, 'config.py'))

train_dataloder = ScanpathDataloder(dataset_name='salicon', phase='train',
                           batch_size=cfg.train_batch_size, input_resize=cfg.input_resize,
                           patch_size=cfg.patch_size, train_gt_num=cfg.train_gt_nums,
                           max_length=cfg.max_length,seed=cfg.seed, train_imgs_num=cfg.train_imgs_num)

train = Trainer(lr=cfg.lr, dataloder=train_dataloder, work_dir=cfg.work_dir, device=cfg.device,
                start_epoch=epoch_start, epoch_nums=cfg.epoch_nums,
                train_gt_num=cfg.train_gt_nums, val_step=cfg.val_step,
                writer=writer, eos=False, lambda_1=0)

evaluation = Evaluation(feature_extrator=feature_extrator, work_dir=cfg.work_dir, writer=writer, val_batch_size=cfg.val_batch_size,
                        device=cfg.device, seed=cfg.seed, max_length=cfg.max_length,
                        action_map_size=cfg.action_map_size, input_resize=cfg.input_resize, patch_size=cfg.patch_size,
                        )

if not cfg.wo_train:
    best_epoch = train.train_epochs(feature_extrator, model, optimizer, lr_scheduler=cfg.lr_scheduler, evaluation=evaluation)
    # load the best performance model
    epoch_start, model, optimizer = loadCheckpoint(model=model, optimizer=optimizer, epoch=best_epoch, work_dir=cfg.work_dir)
else:
    best_epoch = epoch_start

if not cfg.wo_inference:
    # to infer the results on different dataset.
    evaluation.validation(model, best_epoch, dataset_name='osie', save=True, )
    evaluation.validation(model, best_epoch, dataset_name='salicon', save=True, )
    evaluation.validation(model, best_epoch, dataset_name='mit', save=True, )
    evaluation.validation(model, best_epoch, dataset_name='isun', save=True, )

if not cfg.wo_score:
    # to record the score
    resultsToScores(os.path.join(cfg.work_dir, 'seq_results_best_model/'), epoch=best_epoch)





