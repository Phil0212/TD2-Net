# TD<sup>2</sup>-Net: Toward Denoising and Debiasing for Dynamic Scene Graph Generation
This is an official implementation for AAAI 2024 paper ["TD<sup>2</sup>-Net:Toward Denoising and Debiasing for Dynamic Scene Graph Generation".](https://arxiv.org/pdf/2401.12479)

## Overview
**Abstract:** Dynamic scene graph generation (SGG) focuses on detecting objects in a video and determining their pairwise relationships. Existing dynamic SGG methods usually suffer from several issues, including 1) Contextual noise, as some frames might contain occluded and blurred objects. 2) Label bias, primarily due to the high imbalance between a few positive relationship samples and numerous negative ones. Additionally, the distribution of relationships exhibits a long-tailed pattern. To address the above problems, in this paper, we introduce a network named TD<sup>2</sup>-Net that aims at denoising and debiasing for dynamic SGG. Specifically, we first propose a denoising spatio-temporal transformer module that enhances object representation with robust contextual information. This is achieved by designing a differentiable Top-K object selector that utilizes the gumbel-softmax sampling strategy to select the relevant neighborhood for each object.

![GitHub Logo](/data/framework.png)

## Preparation

### Installation
Our codebase is built upon  PyTorch, torchvision, and a few additional dependencies. Please install the packages in the ```environment.yml``` file.

```bash
conda env create -f environment.yml
```


We follow some compiled code for bbox operations.

```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from [here](https://github.com/jwyang/faster-rcnn.pytorch) and download the [pre-trained model](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) for Action Genome dataset. Additionally, place them in
```
fasterRCNN/models/faster_rcnn_ag.pth
```

### Dataset
The Action Genome dataset can be downloaded from [here](https://www.actiongenome.org/#download) . After downloading, please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- ag
    |-- annotations   # gt annotations
    |-- frames        # sampled frames
    |-- videos        # original videos
```
 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```


## Train
+ For PREDCLS: 
```
python train.py -mode predcls -datasize large -log_name $LOGNAME -omega True -TopK 8 -loss ar
```

+ For SGCLS: 
```
python train.py -mode sgcls -datasize large -log_name $LOGNAME -omega True -TopK 8 -loss ar
```
+ For SGDET: 
```
python train.py -mode sgdet -datasize large -log_name $LOGNAME -omega True -TopK 8 -loss ar
```

## Evaluation
<!-- [Trained Models]-->

+ For PREDCLS: 
```
python test.py -mode predcls -datasize large -ckpt $CKPT_PATH$ -TopK 8   
```

+ For SGCLS: 
```
python test.py -mode sgcls -datasize large -ckpt $CKPT_PATH$ -TopK 8  
```
+ For SGDET: 
```
python test.py -mode sgdet -datasize large -ckpt $CKPT_PATH$ -TopK 8
```

## Citing
Please consider citing our paper if it helps your research.
```bash
@inproceedings{lin2024td2,
  title={Td$^2$-net: Toward denoising and debiasing for video scene graph generation},
  author={Lin, Xin and Shi, Chong and Zhan, Yibing and Yang, Zuopeng and Wu, Yaqi and Tao, Dacheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3495--3503},
  year={2024}
}
```

## Acknowledgments 
We gratefully acknowledge the authors of the following repositories, from which portions of our code are adapted.
+ [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch)
+ [Zellers' repository](https://github.com/rowanz/neural-motifs) 
+ [Cong's repository](https://github.com/yrcong/STTran.git)

