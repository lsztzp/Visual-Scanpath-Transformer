work_dir = ''

input_resize = (192, 256)
action_map_size = (192, 256)

feature_dim = 576  # 64  192  576 1472
patch_size = (8, 8)

num_patch_h = input_resize[0] // patch_size[0]
num_patch_w = input_resize[1] // patch_size[1]
num_patchs = num_patch_h * num_patch_w

max_length = 18

# --------------------------------------------------------------------------------------------
#  MODEL PARAMETER
# --------------------------------------------------------------------------------------------

d_model = 128  # Embedding Size
d_ff = 128  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
enc_n_layers = 4  # number of Encoder of Decoder Layer
dec_n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
dropout = 0
num_gauss = 5
MDN_hidden_num = 16
postion_method = 'fixed'

# --------------------------------------------------------------------------------------------
#  Train Setting
# --------------------------------------------------------------------------------------------
device = "cuda:0"
lr = 1e-3
epoch_nums = 20
train_batch_size = 30
val_batch_size = 30

val_step = 5
seed = 1218

weight_decay = 0
lr_scheduler = dict(type='MultiStepLR', warmup_epochs = 20, milestones=[50, 100], gamma=0.5)
train_gt_nums = 1

reload_path = ""

# --------------------------------------------------------------------------------------------
#  test Setting
# --------------------------------------------------------------------------------------------
# sal_gen = False # 显著图是否由 backbone 直接生成
sal_gen = True

backbone_grad = False
backbone_lr = 1e-5 # backbone 梯度更新的学习率

eos = False
lambda_1 = 0.3
# --------------------------------------------------------------------------------------------
#  Ablation Setting
# --------------------------------------------------------------------------------------------
saliency_attention = True # 是否使用显著图进行注意力重加权
vgg_fea = False # 使用 vgg 特征提取
train_imgs_num = 10000 # 训练样本数
