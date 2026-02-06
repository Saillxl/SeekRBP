# SeekRBP


📄 Published in ****  
🔗 [Paper Link](https://ieeexplore.ieee.org/document/11129883)  

---

## 📖 Introduction

![framework](./framework.png)

## Introduction

Bacteriophages are the most abundant and genetically diverse biological entities on Earth, playing a fundamental role in regulating bacterial populations and shaping microbial communities across diverse ecosystems. Many of their ecological and applied functions—including phage therapy, biocontrol, and microbiome engineering—are mediated through highly specific interactions with bacterial hosts. These interactions are encoded in phage genomes, which contain a vast repertoire of proteins with antimicrobial and biotechnological potential. However, the majority of phage-encoded proteins remain poorly characterized, creating a major bottleneck for the effective use of phages in therapeutic, industrial, and agricultural applications, particularly in the era of rising antimicrobial resistance.

Among phage proteins, **receptor-binding proteins (RBPs)** play a central role by mediating the initial adsorption step of infection through specific recognition of host surface receptors. As key determinants of host specificity and infection range, RBPs are critical for applications such as host prediction, rational phage therapy design, and phage engineering. Nevertheless, accurate RBP identification remains challenging due to extreme sequence diversity, rapid evolutionary turnover driven by phage–host co-evolution, and severe class imbalance in phage genomes, where RBPs constitute only a small fraction of encoded proteins.

**SeekRBP** addresses these challenges by reframing RBP identification as a *sequential decision-making problem* rather than a static classification task. The framework integrates sequence-based and structure-based protein representations with a reinforcement learning–inspired adaptive sampling strategy that dynamically prioritizes informative negative samples during training. By coupling multimodal representation learning with adaptive negative sampling, SeekRBP achieves improved recall and robustness in highly imbalanced settings, enabling more reliable large-scale RBP discovery.

### Key Innovations

- **Sequential decision formulation for RBP identification**  
  SeekRBP models negative sample selection as a dynamic decision-making process, explicitly accounting for the evolving informativeness of negative samples during training.

- **Bandit-based adaptive negative sampling**  
  A multi-armed bandit strategy is employed to balance exploration and exploitation, allowing the model to focus on hard and informative negatives without exhaustive sampling.

- **Multimodal sequence–structure integration**  
  SeekRBP jointly leverages one-dimensional sequence representations and three-dimensional structural features to better capture functional determinants of receptor binding.

- **Adaptive expert fusion mechanism**  
  A dedicated fusion module combines additive and multiplicative interactions between sequence and structure features, enabling flexible and expressive cross-modal integration.

- **Scalable and practical design**  
  The framework is computationally efficient and applicable to large-scale phage datasets, making it suitable for real-world phage annotation and downstream host–phage interaction analyses.


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
