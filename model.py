import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmaction.models.backbones import ResNet3d
from loss import AMSoftmax


class Model(torch.nn.Module):
    def __init__(self, feature_dim):
        super(Model, self).__init__()

        # cv模型
        self.cv_model = ResNet3d(depth=18,
                                #   pretrained="", #### required
                                  pretrained=None,
                                  pretrained2d=True,
                                  in_channels=3,
                                  num_stages=4,
                                  base_channels=64,
                                  out_indices=(0, 1, 2, 3, ),
                                  conv1_kernel=(3, 7, 7),
                                  conv1_stride_t=1,
                                  pool1_stride_t=1,
                                  # with_cp=True if "save_memory" in self.conf.keys() else False,
                                  with_pool2=False
                                  # zero_init_residual=False
                                  )
        self.cv_model.pool = nn.AdaptiveAvgPool2d(output_size=[1, 1])
        self.cv_model.fc = nn.Linear(in_features=feature_dim,
                                    out_features=feature_dim)
        # 帧融合
        self.feature_dim = feature_dim
        # 损失
        self.criterian = AMSoftmax(in_feats=feature_dim, n_classes=143)

    def forward(self, data, label=None):
        
        fis = {} # 字典存结果   

        # 如果是GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data = data.cuda()
            if label is not None:
                label = label.cuda()

        data = data.permute(0, 2, 1, 3, 4) #batch,frames,channel w,h->batch,channel,frames,w,h
        cv_feature1, cv_feature2, cv_feature3, cv_feature4 = self.cv_model(data)
        batch_size, c, t, h, w = cv_feature4.shape
        cv_feature_avg = cv_feature4.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w) # 修改
        cv_feature_avg = self.cv_model.pool(cv_feature_avg)

        cv_feature_avg = cv_feature_avg.view(batch_size, t, c, 1, 1) # 修改
        cv_feature_prefc = cv_feature_avg.squeeze(3).squeeze(3)#.permute(0, 2, 1) # 修改
            
        cv_feature_preGTAP = self.cv_model.fc(cv_feature_prefc)
        cv_feature_preGTAP = cv_feature_preGTAP.view(batch_size, -1, self.feature_dim)
        
        cv_feature = torch.mean(cv_feature_preGTAP, dim=1, keepdim=False)

        fis["rate"] = torch.norm(cv_feature, p=2, dim=1)
        cv_feature = torch.div(cv_feature, torch.norm(cv_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)) # 归一化
        fis["cv_feature"] = cv_feature
        fis["cl_feature"] = [cv_feature1, cv_feature2, cv_feature3, cv_feature4, cv_feature_prefc, cv_feature_preGTAP]

        if label is None:
            return fis
        else:
            loss_, costh = self.criterian(cv_feature, label)
            fis["loss"] = loss_
            fis["costh"] = costh

            return fis
