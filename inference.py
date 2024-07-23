import argparse
from torch import optim
from modules.transformer import Transformer
from modules.vgg import VGG_Gen
import torchvision
from utils.utils import *
from mmcv import Config, DictAction
from modules.salgan import Feature_Extrator
from torchvision.transforms import transforms

parser = argparse.ArgumentParser(description='Inference')
# parser.add_argument('--config', default='./model_weights/best_model_and_config/config.py', help='config.py path')
parser.add_argument('--config', default='config.py', help='config.py path')
parser.add_argument('--checkpoint_path', default='./model_weights/ep85.pth.tar', help='checkpoint path')
parser.add_argument('--in_Dir', default='./demo/in_images/', help='path to input')
parser.add_argument('--out_Dir', default='./demo/out_images/', help='path to ouput')
parser.add_argument('--num_generate', default=1, help='the number to generate for each image')
parser.add_argument('--device', default='cuda:0', help='cuda:n')
parser.add_argument('--plot', default=True, help="if plot the results")
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

print(f"Loading {cfg.checkpoint_path}...")
checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()


if __name__=="__main__":
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])
    hw_t = get_hw_t(cfg.action_map_size[0], cfg.action_map_size[1]).to(cfg.device)

    if not os.path.exists(args.out_Dir):
        os.makedirs(args.out_Dir)
    for filename in os.listdir(args.in_Dir):
        print(filename)
        path = args.in_Dir + filename

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
        scanpaths = torch.zeros((args.num_generate, 18, 2))
        for i in range(args.num_generate):    #scanpath number to generate
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

        if args.plot:
            save_path = ""
            # save_path = args.out_Dir + filename.split('.')[0] + '.jpg'
            for i in range(args.num_generate):
                plot_scanpaths(scanpaths[i], img_path=str(path), save_path=save_path,img_height=192,img_witdth=256)
        else:
            save_path = args.out_Dir + filename.split('.')[0] + '.pck'
            torch.save(scanpaths, save_path)

    print('done')

