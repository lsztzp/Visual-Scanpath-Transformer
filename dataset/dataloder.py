from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
from torchvision.transforms import transforms
from utils.utils import *
from pathlib import Path

default_datasets_dir = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset")

paths = {"salicon_test_imgspath": str(default_datasets_dir / 'SALICON/images/all/'),
         "salicon_train_imgspath": str(default_datasets_dir / 'SALICON/images/train/'),
         "salicon_val_imgspath": str(default_datasets_dir / 'SALICON/images/val_5000/'),
         "salicon_salspath": str(default_datasets_dir / 'SALICON/saliency_maps/SalGAN/'),
         "salicon_gtspath": str(default_datasets_dir / 'SALICON/gt_fixations_test/'),

         "osie_test_imgspath": str(default_datasets_dir / 'OSIE/images/all/'),
         "osie_train_imgspath": str(default_datasets_dir / 'OSIE/images/train_new/'),
         "osie_val_imgspath": str(default_datasets_dir / 'OSIE/images/val_new/'),
         "osie_salspath": str(default_datasets_dir / 'OSIE/saliency_maps/SalGAN/'),
         "osie_gtspath": str(default_datasets_dir / 'OSIE/gt_fixations_test/'),

         "mit_test_imgspath": str(default_datasets_dir / 'MIT/images/all/'),
         "mit_salspath": str(default_datasets_dir / 'MIT/saliency_maps/SalGAN/'),
         "mit_gtspath": str(default_datasets_dir / 'MIT/gt_fixations/'),

        "isun_test_imgspath": str(default_datasets_dir / 'iSUN/images/'),
        "isun_gtspath" : str(default_datasets_dir / "iSUN/gt_fixations/"),
        "isun_salspath" : str(default_datasets_dir / "iSUN/saliency_maps/SalGAN/")
         }


class ScanPathDataset(Dataset):
    def __init__(self, phase, dataset_name, input_resize, patch_size=(16, 16), train_gt_num=1, max_length=18, train_imgs_num=10000):
        self.train_imgs_num = train_imgs_num

        self.phase = phase
        self.dataset_name = dataset_name
        self.input_resize = input_resize
        self.train_gt_num = train_gt_num
        self.max_length = max_length
        self.pool2d = nn.AvgPool2d(patch_size)
        self.transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])
        self.salspath = paths[self.dataset_name + "_salspath"]
        self.gtspath = paths[self.dataset_name + "_gtspath"]

        self.imgspath = paths[self.dataset_name + "_" + self.phase + "_imgspath"]
        self.imgspathdir = os.listdir(self.imgspath)
        self.imgspathdir.sort()
        self.imgspathdir = self.imgspathdir[:self.train_imgs_num]

    def __getitem__(self, index):
        image_name = self.imgspathdir[index]
        img = Image.open(os.path.join(self.imgspath, image_name)).convert('RGB')

        img_size = torch.Tensor([img.height, img.width])
        # img, pad_size = pad_img(img, target_w=self.input_resize[1], target_h=self.input_resize[0])
        img = img.resize((self.input_resize[1], self.input_resize[0]))
        img = self.transform(img)

        sal_name = image_name
        sal_path = os.path.join(self.salspath, sal_name)
        sal = Image.open(sal_path)
        # sal, _ = pad_sal(sal, target_w=self.input_resize[1], target_h=self.input_resize[0])
        sal = sal.resize((self.input_resize[1], self.input_resize[0]))
        sal = self.transform(sal).squeeze()

        gt_name = image_name[:-4] + '.mat'

        gt_path = os.path.join(self.gtspath, gt_name)
        ground_truths = scio.loadmat(gt_path)
        gt_fixations = ground_truths['gt_fixations'][0]
        durations_all = torch.zeros(self.train_gt_num, self.max_length)
        if self.dataset_name.upper() == "OSIE":
            durtions = ground_truths['durations'][0]
            for n in range(self.train_gt_num):
                duration = durtions[n]
                if len(duration) == 1:
                    duration = duration[0]
                if len(duration) <= self.max_length:
                    durations_all[n][:len(duration)] = torch.from_numpy(duration.astype(np.float))
                else:
                    durations_all[n] = torch.from_numpy(duration[:self.max_length].astype(np.float))
            durations_all = durations_all / 1000
        if len(gt_fixations) > 10:
            gt_fixations = gt_fixations[:10]
        lens = [len(gt) for gt in gt_fixations]
        avg_len = np.array(lens).mean()
        # score = scio.loadmat(gt_path)['scores'][0]

        gt_fixations_all = torch.zeros(self.train_gt_num, self.max_length, 2)
        valid_lens = torch.zeros(self.train_gt_num).long()
        for n in range(self.train_gt_num):
            gt_fixation = gt_fixations[n]
            valid_lens[n] = len(gt_fixation)
            if len(gt_fixation) <= self.max_length:
                gt_fixations_all[n][:len(gt_fixation)] = torch.from_numpy(gt_fixation.astype(np.float))
            else:
                gt_fixations_all[n] = torch.from_numpy(gt_fixation[:self.max_length].astype(np.float))
        gt_fixations_all[:, :, 0] /= img_size[0]
        gt_fixations_all[:, :, 1] /= img_size[1]

        dec_input = torch.zeros(self.train_gt_num, self.max_length, 2)
        dec_input[:, 1:] = gt_fixations_all[:, :-1]
        dec_input[:, 0, :] = 0.5

        dec_mask = torch.zeros(self.train_gt_num, self.max_length)
        for i in range(self.train_gt_num):
            dec_mask[i, valid_lens[i]:] = 1

        return {
            'imgs': img,
            'sals': sal,
            'gt_fixations': gt_fixations_all,
            "durations": durations_all,
            'valid_lens': valid_lens,
            'dec_inputs': dec_input,
            'dec_masks': dec_mask,
            'file_names': gt_name[:-4],
            'avg_lens': avg_len,
            'img_sizes': img_size
        }

    def __len__(self):
        return len(self.imgspathdir)


class ScanpathDataloder(DataLoader):
    def __init__(self, dataset_name, phase, batch_size,
                 input_resize, patch_size, max_length, train_gt_num=1, seed=1218, train_imgs_num=10000):
        self.seed = seed
        self.train_imgs_num = train_imgs_num
        self.dataset = ScanPathDataset(phase=phase, dataset_name=dataset_name,
                                       input_resize=input_resize, patch_size=patch_size,
                                       train_gt_num=train_gt_num, max_length=max_length, train_imgs_num=self.train_imgs_num)

        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=True,
                         num_workers=16, drop_last=phase == "train", pin_memory=True,
                                         worker_init_fn=self._init_fn)


    def _init_fn(self, worker_id):
        np.random.seed(int(self.seed) + worker_id)
