# Visual ScanPath Transformer: Guiding Computers to See the World [ISMAR2023]

## Trainning
1. To reproduce the training and validation dataset, please referring to [dataloder.py](./dataset/dataloder.py) for placing your dataset files.
2. Execute
```
python main.py
```


## Test
```
python inference.py --in_Dir=./demo/in_images/ --out_Dir=./demo/out_images/
```


## Installation
```
conda create -n  VSPT python=3.9
conda activate VSPT
bash install.sh
```


## Reference
If you find the code useful in your research, please consider citing the paper. 
```
@inproceedings{qiu2023visual,
  title={Visual Scanpath transformer: guiding computers to see the world},
  author={Qiu, Mengyu and Rong, Quan and Liang, Dong and Tu, Huawei},
  booktitle={2023 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  pages={223--232},
  year={2023},
  organization={IEEE}
}
```
