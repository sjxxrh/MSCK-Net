<h2 align="center">
  MSCK-Net: Multiscale Chinese Knot Convolutional Network for Dim and Small Infrared Ship Detection
</h2>




## 1. Quick start

### Setup

```shell
conda create -n msck python=3.11.9
conda activate msck
pip install -r requirements.txt
```


## 2 Data Preparation

<summary>Custom Dataset</summary>

1. **Organize Images:**
The dataset and model weights can be downloaded from Baidu Netdisk:

- **Download Link:**
- 通过网盘分享的文件：MSCK-DATA 链接: https://pan.baidu.com/s/1ZXgpMAxQVqSW0ZN4hz2fvg
- **Extraction Code:** fxga
- IRShip denotes the integrated original dataset, while IRShip-OA denotes the offline-augmented dataset constructed using the offline augmentation strategy described in this paper.

Please download the dataset and organize it according to the following directory structure.

    ```shell
    dataset/
    ├── JPEGImages/
    │   ├── image1.jpg
    │   ├── image1.jpg
    │   └── ...   
    ├── labels/
    │   ├── image1.txt
    │   ├── image1.txt
    │   └── ...  
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── ...
   train.txt
   val.txt
   test.txt
    ```


## 3. Usage
<details open>
<summary> COCO2017 </summary>

1. Training
```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/IS/msck-n.yml
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/IS/msck-s.yml  --test-only -r msck-s.pth
```


<!-- <summary>5. Inference </summary> -->
3. Inference
```shell
python tools/inference/torch_inf.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
```
</details>

## 4. Citation

```latex
@article{MSCK,
  author={Lin, Yuhao and Peng, Dongliang and Wang, Liang and Jiang, Lingjie and Nam, Haewoon},
  title={MSCK-Net: Multiscale Chinese Knot Convolutional Network for Dim and Small Infrared Ship Detection}, 
  journal={IEEE Trans. Geosci. Remote Sens.}, 
  year={2026},
  volume={64},
  number={},
  pages={1-18},
  }
```
Since this dataset is derived from the following datasets, please refer to them as needed.
- NUDT-SIRST-Sea：
- https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection
- ISDD
- https://github.com/yaqihan-9898
- IRSDSS
- https://github.com/greekinRoma/SMPISD-MTPNet
- Maritime-SIRST
- https://github.com/peerless66/Maritime-SIRST

## 5. Acknowledgement
Our work is built upon [D-FINE](https://github.com/Peterande/D-FINE) , [DEIM](https://github.com/Intellindust-AI-Lab/DEIMv2) and [DEIMv2](https://www.shihuahuang.cn/DEIM/).


