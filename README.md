# 🔬 FocalTransNet: A Hybrid Focal-Enhanced Transformer Network for Medical Image Segmentation

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIP%20\(accepted\)-blue)](#-citation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-informational)](#-installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1+-red)](#-installation)

<p align="center">
  <img src="figs\Overview.jpg" width="100%" alt="FocalTransNet overview">
</p>

---

## ✨  Innovations

**FocalTransNet** is a hybrid CNN–Transformer segmentation network designed for high-precision medical image segmentation. It features:
* **Focal-Enhanced (FE) Transformer**: a dual-path module that fuses global self-attention and local convolution with dense cross-connections.
* **Symmetric Patch Merging (SPM)**: a downsampling module with an information-compensation mechanism to preserve fine-grained details (edges, thin structures) during downsampling.

<br>

<p align="center">
  <img src="figs\Innovations.png" width="100%" alt="FocalTransNet overview">
</p>

---

## ⚙️ Installation

```bash
git clone https://github.com/YourUserName/FocalTransNet.git
cd FocalTransNet

# 1) Create environment (example)
conda create -n focaltransnet python=3.9 -y
conda activate focaltransnet

# 2) Install dependencies (strict versions)
pip install -r requirements.txt
```


---

## 📂 Datasets — Download & Prepare

By default we expect all datasets under `./data`. You can change paths in `config.py`. The training scripts will use the dataset-specific sections there.

- **PDGM:**
You can download the preprocessed dataset from [BaiduNetdisk](https://pan.baidu.com/s/1qIoGpvXzNvDP-bct24gFFg?pwd=6ab9), and move into './data/PDGM' folder. 

- **Synapse:**
You can download the preprocessed dataset from [BaiduNetdisk](https://pan.baidu.com/s/1uSRgxyPH_2cN3MSfjGhJJw?pwd=s4v7) or [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd), and move into './data/Synapse' folder. 

- **SegPC2021:**
You can download the preprocessed dataset from [BaiduNetdisk](https://pan.baidu.com/s/1wM435DgkjNZUs280YCRBVA?pwd=3vy2), and move into './data/SegPC2021' folder. 

- **ISIC2018:**
You can download the preprocessed dataset from [BaiduNetdisk](https://pan.baidu.com/s/1fRIsZ0gKadCTNHqyF2OBvg?pwd=hfpc), and move into './data/ISIC2018' folder. 


---

## 🚀 Quick Start — Training & Evaluation

All hyper-parameters and dataset paths are centralized in `config.py`. The evaluation is automatically executed after training.

### PDGM

```bash
python train_PDGM.py 
```

### Synapse

```bash
python train_synapse.py 
```

### ISIC 2018

```bash
python train_ISIC2018.py 
```

### SegPC 2021

```bash
python train_SegPC2021.py 
```

---





## 📜 Citation

If you use this code or find it helpful, please cite:

```bibtex
@article{liao2025focaltransnet,
  title   = {FocalTransNet: A Hybrid Focal-Enhanced Transformer Network for Medical Image Segmentation},
  author  = {Liao, Miao and Yang, Ruixin and Zhao, Yuqian and Liang, Wei and Yuan, Junsong},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025},
  doi     = {10.1109/TIP.2025.3602739}
}
```

---

## 🙏 Acknowledgements
This work was supported in part by the National Natural Science Foundation of China, the Science and Technology Innovation Program of Hunan Province, and the Scientific Research Fund of the Hunan Provincial Education Department.
We thank the authors of  [TransUNet](https://github.com/Beckschen/TransUNet), [CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer), and [CASCADE](https://github.com/SLDGroup/CASCADE) for releasing their code.

---

## 📄 License
This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---
