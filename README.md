This repository holds the PyTorch code for the paper

**A Generalizable Causal-Invariance-Driven Segmentation Model for Peripancreatic Vessels**

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Medical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List

Wenli Fu#, Huijun Hu#, Xinyue Li, Rui Guo, Tao Chen, Xiaohua Qian*

# Abstract

Segmenting peripancreatic vessels in CT, including the superior mesenteric artery (SMA), the coeliac artery (CA), and the partial portal venous system (PPVS), is crucial for preoperative resectability analysis in pancreatic cancer. However, the clinical applicability of vessel segmentation methods is impeded by the low generalizability on multi-center data, mainly attributed to causal-invariance-driven generalizable segmentation model for peripancreatic vessels. It incorporates interventions at both image and feature levels to guide the model to capture causal information by enforcing consistency across datasets, thus enhancing the generalization performance. Specifically, firstly, a contrastdriven image intervention strategy is proposed to construct image-level interventions by generating images with various contrast-related appearances and seeking invariant causal features. Secondly, the feature intervention strategy is designed, where various patterns of feature bias across different centers are simulated to pursue invariant prediction. The proposed model achieved high DSC scores (79.69%, 82.62%, and 83.10%) for the three vessels on a cross-validation set containing 134 cases. Its generalizability was further confirmed on three independent test sets of 233 cases. Overall, the proposed method provides an accurate and generalizable segmentation model for peripancreatic vessels and offers a promising paradigm for increasing the generalizability of segmentation models from a causality perspective. Our source codes will be released at https://github.com/SJTUBME-QianLab/PC_VesselSeg

# Requied

Our code is based on **Python3.7 There are a few dependencies to run the code. The major libraries we depend are

\- PyTorch1.10.0 (http://pytorch.org/)
\- numpy 
\- tqdm 

# Set up

```
pip install -r requirements.txt
```

# Train and test

## Prepare

Firstly, training the contrast-related image generator run  ```./DLOW/train.py``` 

Then, run the  ```./DLOW/tesh_sh.py```  to generate contrast_related perturbed image. The perturbed images will be stored in  ```./data/perturbed_data``` 

## Train

Run the ```Train.py``` by this command:

```
python Train.py
```

After training, the weights will be saved in ```./checkpoint``` folder

## Test

Run the ```Test_6_metrics.py``` by this command:

The evaluation results will be saved in the corresponding ```./checkpoint``` folder as  ```patient_allmetircs_plus.txt``` file including DSC HD, ASD, MSD(MCD), SR (skeleton recall) and SP(skeleton precision).

# Citation

```
@inproceedings{
  title     = {A Generalizable Causal-Invariance-Driven Segmentation Model for Peripancreatic Vessels},
  author    = {Wenli Fu#, Huijun Hu#, Xinyue Li, Rui Guo, Tao Chen, Xiaohua Qian*},
  month     = {March}，
  year      = {2024},
}
```



# Contact

For any question, feel free to contact

```
Wenli Fu : LilyFu@sjtu.edu.cn
```



