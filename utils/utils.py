import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os
from PIL import Image
import torch
import cv2
import matplotlib.cm as cm
import matplotlib
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
    ExponentialLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler

# 按比例填充图片
def pad_sal(sal, target_w, target_h, pad_value=0):
    w, h = sal.size
    if w / h == target_w / target_h:
        return sal, np.array([0, 0])
    if w < h or w * (target_h / target_w) < h:
        new_w = int(h * (target_w / target_h))
        new_img = Image.new('L', (new_w, h), color=pad_value)
        new_img.paste(sal, (int((new_w - w) // 2), 0))
        return new_img, np.array([0, new_w - w])
    else:  #
        new_h = int(w * (target_h / target_w))
        new_img = Image.new('L', (w, new_h), color=pad_value)
        new_img.paste(sal, (0, int((new_h - h) // 2)))
        return new_img, np.array([new_h - h, 0])


# 按比例填充图片
def pad_img(img, target_w, target_h, pad_value=(124, 116, 104)):
    w, h = img.size
    if w / h == target_w / target_h:
        return img, np.array([0, 0])

    if w < h or w * (target_h / target_w) < h:
        new_w = int(h * (target_w / target_h))
        new_img = Image.new('RGB', (new_w, h), color=pad_value)
        new_img.paste(img, (int((new_w - w) // 2), 0))
        return new_img, np.array([0, new_w - w])
    else:  #
        new_h = int(w * (target_h / target_w))
        new_img = Image.new('RGB', (w, new_h), color=pad_value)
        new_img.paste(img, (0, int((new_h - h) // 2)))
        return new_img, np.array([new_h - h, 0])


def show_tensor_heatmap(img, annot=None, fmt=".1f", save_path=None):
    plt.figure(figsize=(10, 20))  # 画布大小
    sns.set()
    ax = sns.heatmap(img, cmap="rainbow", annot=annot, fmt=fmt)  # cmap是热力图颜色的参数

    # plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def show_img_seq(img, seq, save_path=None):
    plt.imshow(img)
    for i in range(len(seq)):
        if i < len(seq) - 1:
            plt.plot([seq[i][1], seq[i + 1][1]], [seq[i][0], seq[i + 1][0]], linewidth=6, alpha=0.7, color="royalblue")

    for i in range(0, len(seq)):
        if i == 0:
            color = 'steelblue'
        elif i == len(seq) - 1:
            color = 'brown'
        else:
            color = 'w'
        plt.scatter(seq[i][1], seq[i][0], s=1200, c=color, alpha=0.7, linewidths=[1], edgecolors="k")  #
        plt.text(seq[i][1], seq[i][0], i + 1, ha='center', va='center', fontsize=18, color="k")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    plt.axis('off')
    return plt

def save_str_file(save_path, str0):
    filename = open(save_path, 'w')
    filename.write(str0)
    filename.close()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)


def save_checkpoint(epoch_num, model, optimizer, work_dir):
    checkpointName = 'ep{}.pth.tar'.format(epoch_num)
    checkpointpath = f'{work_dir}/checkpoint/'
    if not os.path.exists(checkpointpath):
        os.makedirs(checkpointpath)
    checkpoint = {
        'epoch': epoch_num,
        'model': model.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, os.path.join(checkpointpath, checkpointName))


def loadCheckpoint(model, optimizer, work_dir="", epoch=-1, checkpointPath=""):
    reload = False

    if not checkpointPath:
        assert work_dir
        model_dir_name = f'{work_dir}/checkpoint/'
        if not os.path.exists(model_dir_name):
            os.mkdir(model_dir_name)

        model_dir = os.listdir(model_dir_name)  # 列出文件夹下文件名
        if len(model_dir) == 0:
            return 0, model, optimizer
        model_dir.sort(key=lambda x: int(x[2:-8]))  # 文件名按数字排序
        if epoch == -1:
            checkpointName = model_dir[-1]  # 获取文件 , -1 获取最后一个文件
        else:
            checkpointName = 'ep{}.pth.tar'.format(epoch)  # 获取文件 , -1 获取最后一个文件
        checkpointPath = os.path.join(model_dir_name, checkpointName)
        reload = True

    if os.path.isfile(checkpointPath):
        print(f"Loading {checkpointPath}...")
        checkpoint = torch.load(checkpointPath, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.param_groups[0]['lr'] = checkpoint['lr']
        print('Checkpoint loaded')
    else:
        raise OSError('Checkpoint not found')

    if reload:
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    return epoch, model, optimizer


def build_scheduler(optimizer, lr_scheduler):
    name_scheduler = lr_scheduler.type
    scheduler = None

    if name_scheduler == 'StepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = StepLR(optimizer=optimizer, step_size=lr_scheduler.step_size, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=lr_scheduler.T_max)
    elif name_scheduler == 'ReduceLROnPlateau':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step(val_loss)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=lr_scheduler.mode)
    elif name_scheduler == 'LambdaLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler.lr_lambda)
    elif name_scheduler == 'MultiStepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = MultiStepLR(optimizer=optimizer, milestones=lr_scheduler.milestones, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CyclicLR':
        # >>> for epoch in range(10):
        # >>>   for batch in data_loader:
        # >>>       train_batch(...)
        # >>>       scheduler.step()
        scheduler = CyclicLR(optimizer=optimizer, base_lr=lr_scheduler.base_lr, max_lr=lr_scheduler.max_lr)
    elif name_scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingWarmRestarts':
        # >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
        # >>> for epoch in range(20):
        #     >>> scheduler.step()
        # >>> scheduler.step(26)
        # >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=lr_scheduler.T_0,
                                                T_mult=lr_scheduler.T_mult)

    if lr_scheduler.warmup_epochs != 0:
        scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1, total_epoch=lr_scheduler.warmup_epochs, after_scheduler=scheduler)

    if scheduler is None:
        raise Exception('scheduler is wrong')
    return scheduler


def get_hw_t(action_map_h, action_map_w):
    h, w = np.mgrid[0:action_map_h, 0:action_map_w]
    w_t = torch.from_numpy(w / float(action_map_w)).float().reshape(1, 1, -1)
    h_t = torch.from_numpy(h / float(action_map_h)).float().reshape(1, 1, -1)
    hw_t = torch.cat([h_t, w_t], dim=1)

    return hw_t

def normalize_tensor(tensor, rescale=False, zero_fill=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin

    if zero_fill:
        tensor = torch.where(tensor == 0, tensor.max() * 1e-4, tensor)
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor

def plot_scanpaths(scanpaths, img_path, save_path="", img_height=192, img_witdth=256):
    # Plot predicted scanpaths
    # this code is on the basis of ScanGAN https://github.com/DaniMS-ZGZ/ScanGAN360/

    image = cv2.resize(matplotlib.image.imread(img_path), (img_witdth, img_height))
    plt.imshow(image)
    points_x = scanpaths[:, 1]
    points_y = scanpaths[:, 0]

    colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

    previous_point = None
    for num, x, y, c in zip(range(0, len(points_x)), points_x, points_y, colors):
        x *= img_witdth
        y *= img_height
        markersize = 14.
        linewidth = 6.
        if previous_point is not None:
            plt.plot([x, previous_point[0]], [y, previous_point[1]], color='blue', linewidth=linewidth, alpha=0.35)
        previous_point = [x, y]
        plt.plot(x, y, marker='o', markersize=markersize, color=c, alpha=.8)
    plt.axis('off')
    # plt.show(bbox_inches='tight', pad_inches=-0.1)
    # plt.margins(-0.1, -0.1)
    plt.margins(0, 0)
    if not save_path:
        plt.show(bbox_inches='tight', pad_inches=-0.1)
    else:
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=-0.1, dpi=1000)
    plt.cla()