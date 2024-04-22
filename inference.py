import argparse
from torch import optim
from modules.transformer import Transformer
from modules.vgg import VGG_Gen
import torchvision
import time
from utils.utils import *
from mmcv import Config, DictAction

from modules.salgan import Feature_Extrator
from torchvision.transforms import transforms

parser = argparse.ArgumentParser(description='Train a model or Inference')
parser.add_argument('--config', default='./model_weights/best_model_and_config/config.py', help='config.py path')
parser.add_argument('--work_dir', default='test', help='path to save logs and weights')
parser.add_argument('--device', default='cuda:0', help='cuda:n')
parser.add_argument('--wo_train', action="store_true", help='w/o train the model')
parser.add_argument('--wo_inference', action="store_true", help='w/o inference to scanpath results ?')
parser.add_argument('--wo_score', action="store_true", help='w/o score scanpath results ?')
parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

args = parser.parse_args()
cfg = Config.fromfile(args.config)
cfg.merge_from_dict(vars(args))

if args.options is not None:
    cfg.merge_from_dict(args.options)

setup_seed(cfg.seed)

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


if __name__=="__main__":
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])
    hw_t = get_hw_t(cfg.action_map_size[0], cfg.action_map_size[1]).to(cfg.device)

    input_path="./01-Datasets/in_images/"
    output_path="./01-Datasets/out_images/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for filename in os.listdir(input_path):
        print(filename)
        path = input_path + filename

        img = Image.open(path).convert('RGB')
        img_size = torch.Tensor([img.height, img.width])
        img = img.resize((cfg.input_resize[1], cfg.input_resize[0]))
        img = transform(img)
        imgs = img.unsqueeze(dim=0).to(cfg.device)
        enc_masks = None
        batch_size = len(imgs)

        enc_inputs = feature_extrator(imgs, None)
        enc_outputs, enc_self_attns = model.encoder(enc_inputs, None)
        val_batch_size = 1
        scanpaths = torch.zeros((10, 18, 2))
        for i in range(1):    #scanpath number to generate
            enc_inputs = feature_extrator(imgs, None)
            enc_outputs, enc_self_attns = model.encoder(enc_inputs, None)

            dec_inputs = torch.ones(val_batch_size, 1, 2).to(cfg.device) * 0.5

            for n in range(cfg.max_length):
                dec_outputs, dec_self_attns, dec_enc_attns = model.decoder(enc_outputs, dec_inputs,
                                                                           enc_masks=enc_masks,
                                                                           dec_masks=torch.zeros(val_batch_size,
                                                                                                 n + 1).to(cfg.device))

                pis, mus, sigmas, rhos = model.mdn(dec_outputs)

                pred_roi_maps = model.mdn.mixture_probability_map(pis, mus, sigmas, rhos, hw_t).reshape(
                    (-1, n + 1, cfg.action_map_size[0], cfg.action_map_size[1])).flatten(2)

                _, indexs = pred_roi_maps.max(-1)
                indexs_w = ((indexs % cfg.action_map_size[1]) / cfg.action_map_size[1]).unsqueeze(-1)
                indexs_h = (torch.div(indexs, cfg.action_map_size[1], rounding_mode='trunc') / cfg.action_map_size[
                    0]).unsqueeze(-1)
                outputs = torch.cat((indexs_h, indexs_w), -1)

                outputs = outputs.clamp(0, 0.99)
                last_fixations = outputs[:, -1]
                dec_inputs = torch.cat((dec_inputs, last_fixations.unsqueeze(1)), dim=1)

            scanpaths[i] = dec_inputs[0][1:]

        save_path = output_path + filename.split('.')[0] + '.pck'
        torch.save(scanpaths, save_path)
