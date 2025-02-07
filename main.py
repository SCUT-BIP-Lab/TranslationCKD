# Demo Code for Paper:
# [Title]  - "Learning an Augmented RGB Representation for Dynamic Hand Gesture Authentication"
# [Author] - Huilong Xie, Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/TranslationCKD

import torch
from cl_mfenl import CL_MSCL

def train_demo():
    # the feature dim of last feature map (layer4) from i3d_resnet18 is 512
    feature_dim = 512
    model = CL_MSCL(feature_dim=feature_dim)
    """
    todo: loading the weights of teacher networks

    """
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    for epoch in range(0, 100):
        # the first tensor of the modal dimension is the target modality
        data = torch.randn(2, 64, 3, 2, 224, 224) #batch, frame, channel, modality, h, w
        data = data.view(2, 64, -1, 224, 224) #regard the frame as batch
        label = torch.ones(2) # there are 143 identities in the training set
        fis = model.co_learning(data, label, optimizer)

    return id_feature

def test_demo():
    # the feature dim of last feature map (layer4) from i3d_resnet18 is 512
    feature_dim = 512
    model = CL_MSCL(feature_dim=feature_dim)
    data = torch.randn(1, 64, 3, 1, 224, 224) #batch, frame, channel, modality, h, w
    fis = model(data)
    feature = fis['cv_feature']

if __name__ == '__main__':
    train_demo()
    print("Demo is finished!")