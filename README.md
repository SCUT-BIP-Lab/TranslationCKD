# Knowledge Translation
Pytorch Implementation of paper:

> **Learning an Augmented RGB Representation for Dynamic Hand Gesture Authentication**
>
> Huilong Xie, Wenwei Song, Wenxiong Kang\*.

## Main Contribution
Dynamic hand gesture authentication aims to recognize users' identity through the characteristics of their hand gestures. How to extract favorable features for verification is the key to success. Cross-modal knowledge distillation is an intuitive approach that can introduce additional modality information in the training phase to enhance the target modality representation, improving model performance without incurring additional computation in the inference phase. However, most previous cross-modal knowledge distillation methods directly transfer information from one modality to another one without considering the modality gap. In this paper, we propose a novel translation mechanism in cross-modal knowledge distillation that can effectively mitigate the modality gap and utilize the information from the additional modality to enhance the target modality representation. In order to better transfer modality information, we propose a novel modality fusion-enhanced non-local (MFENL) module, which can fuse the multi-modal information from the teacher network and enhance the fused features based on the modality input into the student network. We use cascaded MFENL modules as the translator based on the proposed cross-modal knowledge distillation method to learn an enhanced RGB representation for dynamic hand gesture authentication. Extensive experiments on the SCUTDHGA dataset demonstrate that our method has compelling advantages over the state-of-the-art methods.

<div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/TranslationCKD/master/img/img1.png" />
</p>
</div>

The proposed knowledge translation method. N is the number of the frame, and the lock in the top right corner of the teacher denotes that the parameters of the teacher networks are frozen when the student network is being trained. All networks share identical structures, and we indicate the output feature shape (B, C, N, H, W ) at each stage. Here, we take as an example that the additional modality are optical flow modality. This is a two-stage knowledge distillation process. First, all modalities, including the RGB modality used in the inference phase, are employed to train their respective networks, and then their parameters are frozen as teacher networks. Next, the features from each teacher network at the same stage are fed into the translator. The translator comprises cascaded MFENL modules that facilitate student learning through translating knowledge transmitted from the teacher into more comprehensible content. The comparison between the translated features and the student features is represented by bi-directional red dashed arrows, where teacher’s knowledge is transferred to the student network by reducing the distance between the two features.

<div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/TranslationCKD/master/img/img2.png" />
</p>
</div>

The details of the MFENL module. ConvD is a 2 × 2 convolution with downsampling for matching the shape. The PM and RS denote permute and reshape operations, respectively. The weights and biases of the Batch Normalization Layer (BNF and BNE) are initialized to zero. Best viewed in color.

## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of knowledge translation. Through debugging ([main.py](/main.py)), you can quickly understand the configuration and building method of [Knowledge Translation](/cl_mfenl.py), including the MFENL module.

If you want to explore the entire hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).
