# SeekRBP


📄 Published in ****  
🔗 [Paper Link](https://ieeexplore.ieee.org/document/11129883)  

---

## 📖 Introduction

![framework](./LGFFM/framework.png)

Accurate segmentation of ultrasound images plays a critical role in disease screening and diagnosis. Recently, neural network-based methods have shown great promise, but still face challenges due to the inherent characteristics of ultrasound images—such as **low resolution, speckle noise, and artifacts**.

Moreover, ultrasound segmentation spans a wide range of scenarios, including **organ segmentation** (e.g., cardiac, fetal head) and **lesion segmentation** (e.g., breast cancer, thyroid nodules), which makes the task highly diverse and complex. Existing methods are often tailored for specific cases, limiting their flexibility and generalization.

To address these challenges, we propose **LGFFM (Localized and Globalized Frequency Fusion Model)**, a novel framework for ultrasound image segmentation:

- **Parallel Bi-Encoder (PBE):** integrates Local Feature Blocks (LFB) and Global Feature Blocks (GLB) to enhance feature extraction.  
- **Frequency Domain Mapping Module (FDMM):** captures texture information, particularly high-frequency details like edges.  
- **Multi-Domain Fusion (MDF):** effectively integrates features across different domains for more robust segmentation.  

We evaluate LGFFM on **eight public ultrasound datasets across four categories**. Results show that LGFFM **outperforms state-of-the-art methods** in both segmentation accuracy and generalization performance.  

---

## 📂 Clone Repository

```bash
git clone https://github.com/Saillxl/SeekRBP.git
cd SeekRBP/
```

---

## 📑 Dataset Preparation

The dataset should follow the format below. 

```


## ⚙️ Requirements
We recommend creating a clean environment:

```
conda create -n LGFFM python=3.10
conda activate LGFFM
pip install -r requirements.txt
```

## 🚀 Training
1. Download pre-trained weights from the official (not 2.1) [SAM2 large repository]. Place the weights in the "checkpoints" folder. If you want to place it elsewhere, modify the parameter hiera_path in train.py.
2. Run train.py
```
python train.py --hiera_path './checkpoints/sam2_hiera_large.pt' --train_image_path 'data/BUSI/train/img.yaml' --train_mask_path 'data/BUSI/train/ann.yaml' --save_path 'output/BUSI' 
```

## 🧪 Testing
Run test.py
```
python test.py --checkpoint 'output/BUSI/SAM2-UNet-70.pth' --test_image_path 'data/BUSI/test/img.yaml' --test_gt_path 'data/BUSI/test/ann.yaml' --save_path 'output/'
```

## 📌 Citation
If you find this repository useful, please cite our paper(bibtex):
```

```

## 🙏 Acknowledgement


Public ultrasound datasets (e.g., BUSI)
##
