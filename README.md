# SeekRBP


📄 Published in ****  
🔗 [Paper Link](https://ieeexplore.ieee.org/document/11129883)  

---

## 📖 Introduction

![framework](./framework.png)

Bacteriophages are the most abundant and genetically diverse biological entities on Earth and play a central role in regulating bacterial populations and shaping microbial communities. Many of their ecological and applied functions, including phage therapy, biocontrol, and microbiome engineering, rely on specific interactions with bacterial hosts. These interactions are encoded in phage genomes, which contain a large number of proteins with antimicrobial and biotechnological potential. However, most phage-encoded proteins remain poorly characterized, creating a major obstacle to the effective use of phages, particularly in the context of rising antimicrobial resistance.

Receptor-binding proteins (RBPs) are key mediators of phage–host interactions, as they initiate infection by recognizing and binding to host surface receptors. RBPs largely determine host specificity and infection range, making their accurate identification essential for applications such as host prediction and rational phage design. Despite their importance, RBP identification is challenging due to extreme sequence diversity, rapid phage–host co-evolution, and severe class imbalance, since RBPs constitute only a small fraction of phage-encoded proteins.

**SeekRBP** addresses these challenges by reframing RBP identification as a *sequential decision-making problem*. The framework integrates sequence-based and structure-based protein representations with a reinforcement learning–inspired adaptive sampling strategy that dynamically prioritizes informative negative samples. This combination improves recall and robustness in highly imbalanced settings, enabling more reliable large-scale RBP discovery.

### Key Innovations

- **Sequential decision perspective**  
  RBP identification is formulated as a dynamic decision-making process that accounts for evolving negative sample informativeness.

- **Bandit-based adaptive sampling**  
  A multi-armed bandit strategy balances exploration and exploitation to focus training on hard and informative negatives.

- **Sequence–structure integration**  
  Both sequence and structural features are jointly leveraged to capture functional determinants of receptor binding.

- **Adaptive feature fusion**  
  An expert fusion mechanism flexibly models interactions between sequence and structure representations.

- **Scalable design**  
  The framework is efficient and suitable for large-scale phage protein annotation and downstream analyses.



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
