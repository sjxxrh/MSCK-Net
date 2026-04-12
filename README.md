<h2 align="center">
  MSCK-Net: Multiscale Chinese Knot Convolutional Network for Dim and Small Infrared Ship Detection
</h2>

1. **NOTE:**
由于将原项目的改进部分迁移到DEIMv2上，可能由于小数四舍五入的问题，个别原IRShip数据集的精度与论文略有差别（0.1%左右），但不影响模型结果。


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

    Structure your dataset directories as follows:

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

## 5. Acknowledgement
Our work is built upon [D-FINE](https://github.com/Peterande/D-FINE) , [DEIM](https://github.com/Intellindust-AI-Lab/DEIMv2) and [DEIMv2](https://www.shihuahuang.cn/DEIM/).


