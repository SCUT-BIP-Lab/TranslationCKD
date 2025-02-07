import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import AMSoftmax
from model import Model

class MFNL_module(nn.Module):
    def __init__(self,  in_channels: int, reduction=2):

        super(MFNL_module, self).__init__()
        self.reduction = reduction
        self.in_channels = in_channels

        self.convQ = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.convK = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.convV = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1)

        self.conv_reconstruct = nn.Sequential(
            nn.Conv2d(in_channels // self.reduction, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        nn.init.constant_(self.conv_reconstruct[1].weight, 0)
        nn.init.constant_(self.conv_reconstruct[1].bias, 0)

    def forward(self, x: torch.Tensor):

        b, c, m, n = x.size() # m: modality, n: hw or thw
        cr = c // self.reduction
        assert c == self.in_channels, 'input channel not equal!'

        Q = self.convQ(x)  # (b, cr, m, n)
        K = self.convK(x)  # (b, cr, m, n)
        V = self.convV(x)  # (b, cr, m, n)

        # for channel-independent modality self-attention
        Q = Q.view(b*cr, m, n) # (b*cr, m, n)
        K = K.view(b*cr, m, n) # (b*cr, m, n)
        V = V.view(b*cr, m, n) # (b*cr, m, n)

        # calculate affinity matrices for each pattern (channel)
        correlation = torch.bmm(Q, K.permute(0, 2, 1))  # (b*cr, m, m)
        correlation_attention = F.softmax(correlation, dim=-1)
        # global modality information aggregated for each channel
        y = torch.matmul(correlation_attention, V) # (b*cr, m, n)
        y = y.view(b, cr, m, n) # (b, cr, m, n)
        y = self.conv_reconstruct(y) # (b, c, m, n)
        z = y + x
        return z
    
class MENL_module(nn.Module):
    def __init__(self,  in_channels: int, reduction=2):

        super(MENL_module, self).__init__()
        self.reduction = reduction
        self.in_channels = in_channels

        self.convQ = nn.Conv1d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.convK = nn.Conv1d(2*in_channels, 2*in_channels // self.reduction, kernel_size=1)
        self.convV = nn.Conv1d(2*in_channels, 2*in_channels // self.reduction, kernel_size=1)

        self.conv_reconstruct = nn.Sequential(
            nn.Conv1d(in_channels // self.reduction, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels)
        )

        nn.init.constant_(self.conv_reconstruct[1].weight, 0)
        nn.init.constant_(self.conv_reconstruct[1].bias, 0)

    def forward(self, x: torch.Tensor, x_fuse: torch.Tensor):

        b, c, n = x.size() # n: hw or thw
        cr = c // self.reduction
        assert c == self.in_channels, 'input channel not equal!'

        Q = self.convQ(x)  # (b, cr, n)
        K = self.convK(x_fuse)  # (b, 2*cr, n)
        V = self.convV(x_fuse)  # (b, 2*cr, n)

        # calculate affinity matrices
        correlation = torch.bmm(Q, K.permute(0, 2, 1))  # (b, cr, 2*cr)
        correlation_attention = F.softmax(correlation, dim=-1)
        # global modality information aggregated for each channel
        y = torch.matmul(correlation_attention, V) # (b, cr, n)
        y = self.conv_reconstruct(y) # (b, c, n)
        z = y + x
        return z
    
class MFENLWapper(nn.Module):
    def __init__(self, in_channels):
        super(MFENLWapper, self).__init__()
        self.MFNL = MFNL_module(in_channels)
        self.MENL = MENL_module(in_channels)
        if in_channels != 64:
            self.convp = nn.Conv3d(in_channels//2, in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x_rgb, x_flow, z_=None):
        b, c, t, h, w = x_rgb.size()
        x_rgb_ = x_rgb
        x_flow_ = x_flow
        if z_ is not None:
            if c != 64:
                z_p = self.convp(z_)
            x_rgb_ = x_rgb_ + z_p
            x_flow_ = x_flow_ + z_p
        x_rgb_ = x_rgb_.unsqueeze(2)
        x_flow_ = x_flow_.unsqueeze(2)
        x = torch.cat((x_rgb_, x_flow_), dim=2) # b, c, 2, t, h, w
        x = x.permute(0, 3, 1, 2, 4, 5).reshape(b*t, c, 2, h*w)
        y = self.MFNL(x) # b, c, 2, n
        y = y.view(b*t, c*2, h*w)
        x_ = x_rgb.permute(0, 2, 1, 3, 4).reshape(b*t, c, h*w)
        z = self.MENL(x_, y)
        z = z.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, t, h, w)

        return z

class cfuse_layer(nn.Module):
    def __init__(self):
        # start 0, 1, 2, 3 -> ResNet 4个 stage
        super(cfuse_layer, self).__init__()
        assert start > -1
        assert start < 4
        self.start = start
        self.method = method
        
        self.fuse_layer1 = MFENLWapper(128, aggregation)
        self.fuse_layer2 = MFENLWapper(256, aggregation)
        self.fuse_layer3 = MFENLWapper(512, aggregation)

    def forward(self, x_rgb, x_flow):
        x_fuse_list = []
        x_fuse_list.append(self.fuse_layer1(x_rgb[1], x_flow[1], x_fuse_list[-1] if self.start < 1 else None))
        x_fuse_list.append(self.fuse_layer2(x_rgb[2], x_flow[2], x_fuse_list[-1] if self.start < 2 else None))
        x_fuse_list.append(self.fuse_layer3(x_rgb[3], x_flow[3], x_fuse_list[-1] if self.start < 3 else None))

        return x_fuse_list

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, t, h, w = fs.shape
        loss = F.l1_loss(fs, ft.detach(), reduction='mean')
        cnt = 1.0
        tot = 1.0
        # for l in [32, 16, 8, 4, 2, 1]:
        #     if l >= t:
        #         continue
        #     tmpfs = F.adaptive_avg_pool3d(fs, (l,h,w))
        #     tmpft = F.adaptive_avg_pool3d(ft, (l,h,w))
        #     cnt /= 2.0
        #     loss += F.mse_loss(tmpfs, tmpft.detach(), reduction='mean') * cnt
        #     tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

class CL_MSCL(torch.nn.Module):
    def __init__(self, feature_dim):
        super(CL_MSCL, self).__init__()
        self.weight_fea = 0.1
        self.weight_logit = 5.0
        self.num_tea = 2
        self.tea_model = nn.ModuleList()
        for i in range(self.num_tea):
            self.tea_model.append(Model(feature_dim))

        self.stu_model = Model(feature_dim)

        self.CL = nn.ModuleList()
        self.CL.add_module("cfuse_layer", cfuse_layer())
        
        self.CL.add_module("fc", nn.Linear(in_features=feature_dim, out_features=feature_dim))
        self.CL.add_module("bn1", nn.BatchNorm1d(feature_dim))
        nn.init.constant_(self.CL.bn1.weight, 0)
        nn.init.constant_(self.CL.bn1.bias, 0)

        self.CL.add_module("criterian", AMSoftmax(in_feats=2*feature_dim, n_classes=143))

        self.CL.add_module("cv_match_fea_dim", nn.ModuleList())
        s_dim = feature_dim
        for i in range(3):
            self.CL.cv_match_fea_dim.append(nn.Conv3d(in_channels=s_dim, out_channels=s_dim, kernel_size=1))
            s_dim = s_dim // 2

        self.CL.add_module("cv_match_logit_dim", nn.Linear(in_features=feature_dim, out_features=2*feature_dim))

    def forward(self, data):
        data = data.view(data.shape[:2]+(3, -1)+data.shape[-2:]).permute(3, 0, 1, 2, 4, 5).contiguous()
        stu_fis = self.stu_model(data[0])
        return stu_fis
    
    def co_learning(self, data, label, optimizer):
        data = data.view(data.shape[:2]+(3, -1)+data.shape[-2:]).permute(3, 0, 1, 2, 4, 5).contiguous()
        if label is not None:
            label = label.cuda()
        tea_fis = []
        with torch.no_grad():
            for i in range(self.num_tea):
                tea_fis.append(self.tea_model[i](data[i]))
        
        stu_fis = self.stu_model(data[0], label)

        x_fuse_list = []
        loss_fuse = 0
        
        x_rgb = tea_fis[0]["cl_feature"][:4]
        x_flow = tea_fis[1]["cl_feature"][:4]

        x_fuse_list = self.CL.cfuse_layer(x_rgb, x_flow)

        # x_fuse = F.adaptive_avg_pool3d(x_fuse_list[-1], (None, 1, 1))
        # x_fuse = x_fuse.squeeze(3).squeeze(3).permute(0, 2, 1)
        # 修改
        cv_feature4 = x_fuse_list[-1]
        b, c, t, h, w = cv_feature4.shape
        cv_feature4 = cv_feature4.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        x_fuse = F.adaptive_avg_pool2d(cv_feature4, (1, 1))
        x_fuse = x_fuse.squeeze(2).squeeze(2).view(b, t, c)
        
        cv_feature_tfuse = self.CL.fc(x_fuse)
        cv_feature_tfuse = cv_feature_tfuse.permute(0, 2, 1).contiguous()
        cv_feature_tfuse = self.CL.bn1(cv_feature_tfuse).permute(0, 2, 1).contiguous()
        cv_feature_tfuse = torch.mean(cv_feature_tfuse, dim=1, keepdim=False)
        cv_feature_tfuse = torch.cat((tea_fis[0]["cv_feature"]+cv_feature_tfuse, tea_fis[1]["cv_feature"]+cv_feature_tfuse), dim=-1)
        rate = torch.norm(cv_feature_tfuse, p=2, dim=1, keepdim=True).mean()
        cv_feature_tfuse = torch.div(cv_feature_tfuse, torch.norm(cv_feature_tfuse, p=2, dim=1, keepdim=True).clamp(min=1e-12)) # 归一化
        # 帧间融合()
        stu_fis["cv_feature_tfuse"] = cv_feature_tfuse
        loss_fuse, costh = self.CL.criterian(cv_feature_tfuse, label)
        stu_fis["loss_fuse"] = loss_fuse
        
        stu_feature_list = stu_fis["cl_feature"][4-len(x_fuse_list):4]
        for i in range(len(stu_feature_list)):
            stu_feature_list[i] = self.CL.cv_match_fea_dim[len(self.CL.cv_match_fea_dim)-i-1](stu_feature_list[i])
        
        loss_dis = hcl(stu_feature_list, x_fuse_list)
        stu_fis["loss_dis"] = loss_dis

        loss_stu = stu_fis["loss"]

        stu_cv_feature = stu_fis["cv_feature"]
        stu_cv_feature = self.CL.cv_match_logit_dim(stu_cv_feature)

        loss_logit = rate.detach() * rate.detach() * F.mse_loss(stu_cv_feature, cv_feature_tfuse.detach(), reduction='sum')
        stu_fis["loss_logit"] = loss_logit
        
        optimizer.zero_grad() # 优化器清空
        loss_ = loss_logit * self.weight_logit + loss_dis * self.weight_fea + loss_stu + loss_fuse
        loss_.backward() # 反向
        optimizer.step()

        return stu_fis