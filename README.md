# ID<sup>2</sup>-Net: Toward Denoising and Debiasing for Dynamic Scene Graph Generation
Pytorch Implementation of the framework **ID<sup>2</sup>-Net** proposed in our paper [Toward Denoising and Debiasing for Dynamic Scene Graph Generation](https://arxiv.org/pdf/2401.12479)

## Overview
**Abstract:** Dynamic scene graph generation (SGG) focuses on detecting objects in a video and determining their pairwise relationships. Existing dynamic SGG methods usually suffer from several issues, including 1) Contextual noise, as some frames might contain occluded and blurred objects. 2) Label bias, primarily due to the high imbalance between a few positive relationship samples and numerous negative ones. Additionally, the distribution of relationships exhibits a long-tailed pattern. To address the above problems, in this paper, we introduce a network named TD$^2$-Net that aims at denoising and debiasing for dynamic SGG. Specifically, we first propose a denoising spatio-temporal transformer module that enhances object representation with robust contextual information. This is achieved by designing a differentiable Top-K object selector that utilizes the gumbel-softmax sampling strategy to select the relevant neighborhood for each object.

![GitHub Logo](/data/framework.png)

## Requirements
Please install packages in the ```environment.yml``` file.

## Usage

We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

## Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- ag
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
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
python test.py -mode predcls -datasize large -ckpt $CKPT_PATH -TopK 8   
```

+ For SGCLS: 
```
python test.py -mode sgcls -datasize large -ckpt $CKPT_PATH -TopK 8  
```
+ For SGDET: 
```
python test.py -mode sgdet -datasize large -ckpt $CKPT_PATH -TopK 8
```

## Acknowledgments 
We would like to acknowledge the authors of the following repositories from where we borrowed some code
+ [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch)
+ [Zellers' repository](https://github.com/rowanz/neural-motifs) 
+ [Cong's repository](https://github.com/yrcong/STTran.git)

