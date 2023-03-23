# Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild

## Introduction
* This repo is official **[PyTorch](https://pytorch.org)** implementation of **Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild (CVPR 2023)**. 

<p align="middle">
<img src="assets/teaser.png" width="1200" height="275">
</p>

## Demo
1. Move to `demo` folder.
2. Download pre-trained InterWild from [here](https://drive.google.com/file/d/1W4TC5MAqciG5qN79wtKBGL8mGgqrfEvP/view?usp=share_link).
3. Put input images at `images`. The image should be a cropped image, which contain a single human. For example, using a human detector.
4. Run `python demo.py --gpu $GPU_ID`
5. Boxes, meshes, and MANO parameters are saved at `boxes`, `meshes`, `params`, respectively.

## Directory

### Root
The `${ROOT}` is described as below.
```
${ROOT}
|-- data
|-- demo
|-- common
|-- main
|-- output
```
* `data` contains data loading codes and soft links to images and annotations directories.
* `demo` contains the demo code
* `common` contains kernel codes for 3D interacting hand pose estimation.
* `main` contains high-level codes for training or testing the network.
* `output` contains log, trained models, visualized outputs, and test result.

### Data
You need to follow directory structure of the `data` as below.
```
${ROOT}
|-- data
|   |-- InterHand26M
|   |   |-- annotations
|   |   |   |-- train
|   |   |   |-- test
|   |   |-- images
|   |-- MSCOCO
|   |   |-- annotations
|   |   |   |-- coco_wholebody_train_v1.0.json
|   |   |   |-- coco_wholebody_val_v1.0.json
|   |   |   |-- MSCOCO_train_MANO_NeuralAnnot.json
|   |   |-- images
|   |   |   |-- train2017
|   |   |   |-- val2017
|   |-- HIC
|   |   |-- data
|   |   |   |-- HIC.json
```
* Download InterHand2.6M [[HOMEPAGE](https://mks0601.github.io/InterHand2.6M/)]. `images` contains images in 5 fps, and `annotations` contains the `H+M` subset.
* Download the whole-body version of MSCOCO [[HOMEPAGE](https://github.com/jin-s13/COCO-WholeBody/)]. `MSCOCO_train_MANO_NeuralAnnot.json` can be downloaded from [[here](https://drive.google.com/file/d/1OuWlMor5f0TZLVSsojz5Mh6Ut93WkcJc/view)].
* Download HIC [[HOMEPAGE](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/)] [[annotations](https://drive.google.com/file/d/1oqquzJ7DY728M8zQoCYvvuZEBh8L8zkQ/view?usp=share_link)]. You need to download 1) all `Hand-Hand Interaction` sequences (`01.zip`-`14.zip`) and 2) some of `Hand-Object Interaction` seuqneces (`15.zip`-`21.zip`) and 3) MANO fits. Or you can simply run `python download.py` in the `data/HIC` folder.
* All annotation files follow [MSCOCO format](http://cocodataset.org/#format-data). 
* If you want to add your own dataset, you have to convert it to [MSCOCO format](http://cocodataset.org/#format-data).  

### Output
You need to follow the directory structure of the `output` folder as below.
```
${ROOT}
|-- output
|   |-- log
|   |-- model_dump
|   |-- result
|   |-- vis
```
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* `vis` folder contains visualized results.

## Running InterWild
### Start
* As InterHand2.6M is too large, I only use data whose `ann_id` belong to the `H` subset, while values belong to the `H+M` subset.
* To this end, please download `aid_human_annot_train.txt` and `aid_human_annot_test.txt` from [here](https://drive.google.com/file/d/1Tz6P2pyc55L5ZcGx1W85v5hAnB-MZ8ok/view?usp=share_link) and [here](https://drive.google.com/file/d/1NBGADofWE76ksA2S1bXy-kZSW6w0UIyF/view?usp=share_link), respectively, and place them in `main` folder.

### Train
In the `main` folder, run
```bash
python train.py --gpu 0-3
```
to train the network on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. If you want to continue experiment, run use `--continue`. 


### Test
* If you want to test with pre-trained InterWild, download it from [here](https://drive.google.com/file/d/1W4TC5MAqciG5qN79wtKBGL8mGgqrfEvP/view?usp=share_link) and place it at `output/model_dump'.
* Or if you want to test with our own trained model, place your model at `output/model_dump`.

In the `main` folder, run 
```bash
python test.py --gpu 0-3 --test_epoch 6
```
to test the network on the GPU 0,1,2,3 with `snapshot_6.pth.tar`.  `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. 

## Reference  
```  
@InProceedings{Moon_2023_CVPR_InterWild,  
author = {Moon, Gyeongsik},  
title = {Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild},  
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
year = {2023}  
}  
```

## License
This repo is CC-BY-NC 4.0 licensed, as found in the LICENSE file.

[[Terms of Use](https://opensource.facebook.com/legal/terms)]
[[Privacy Policy](https://opensource.facebook.com/legal/privacy)]

