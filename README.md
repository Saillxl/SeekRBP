# SeekRBP


📄 Published in ****  
🔗 [Paper Link](https://ieeexplore.ieee.org/document/11129883)  

---

## 📖 Introduction

![framework](./framework.png)

## Introduction

Bacteriophages are the most abundant and genetically diverse biological entities on Earth and play a central role in regulating bacterial populations and shaping microbial ecosystems. Many important applications of phages, including phage therapy, biocontrol, and microbiome engineering, rely on their specific interactions with bacterial hosts. These interactions are encoded in phage genomes, which contain a large number of proteins with potential biological and biotechnological value. However, most phage-encoded proteins remain poorly annotated, limiting the effective use of phages, especially under the growing threat of antimicrobial resistance.

Receptor-binding proteins (RBPs) are key mediators of phage–host interactions, as they initiate infection by recognizing and binding bacterial surface receptors. Accurate identification of RBPs is therefore essential for host prediction and downstream phage applications. This task is challenging due to extreme sequence diversity, rapid phage–host co-evolution, and severe class imbalance, where RBPs represent only a small fraction of phage proteins.

**SeekRBP** addresses these challenges by modeling RBP identification as a sequential decision-making problem. The framework combines sequence and structure representations with a reinforcement learning–inspired adaptive sampling strategy to improve recall and robustness in highly imbalanced settings.

### Key Innovations

- **Sequential decision-based formulation** that captures the dynamic informativeness of negative samples during training  
- **Bandit-based adaptive negative sampling** to efficiently focus on hard and informative negatives  
- **Joint sequence–structure modeling** to better represent functional determinants of receptor binding  
- **Adaptive fusion mechanism** for effective cross-modal feature integration  


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
conda create -n SeekRBP python=3.10
conda activate SeekRBP
pip install -r requirements.txt
```

## Sequence Only
### 🚀 Training
1. Extract sequence features using ESM2
2. Run train.py
```
python train.py --hiera_path './checkpoints/sam2_hiera_large.pt' --train_image_path 'data/BUSI/train/img.yaml' --train_mask_path 'data/BUSI/train/ann.yaml' --save_path 'output/BUSI' 
```

### 🧪 Testing
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
