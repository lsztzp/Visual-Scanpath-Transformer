--wo_train: 不进行训练，需要指定 reload_path 加载历史模型
--wo_inference: 默认在几个测试数据集上推理魔模型结果，并存在 workdir 下指定文件夹
--wo_score： 默认读取workdir 下几个测试数据集上模型推理结果计算数据集模型分数，保存在 workdir 下 txt 文件中。


# 模型的加载：
1. 指定 reload_path = ""
2. 默认加载 work_dir 中最后模型（用于断点续训），没有则不加载 [问题：为了记录时间，加入了日期，如果不在同一天训练完，就会有问题]



--config=/data/qmengyu/02-Results/01-ScanPath/logs/04-13-test-baseline-le-1e-3/config.py
--work_dir="04-13-test-baseline-le-1e-3"
--device="cuda:2"
--wo_score




conda create -n  pytorch python=3.9

conda activate pytorch

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install -U openmim
mim install mmcv-full

pip3 install scipy

pip3 install seaborn

pip3 install warmup_scheduler

pip3 install multimatch_gaze

pip3 install dtaidistance

pip3 install tensorboard

pip3 install einops

pip install "numpy<1.24"