"""

Contributed by Wenbin Li & Jinglin Xu

"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
import scipy.io as scio

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_MultiViewNet(pretrained=False, model_root=None, which_model='multiviewNet', norm='batch', init_type='normal',
                        use_gpu=True, num_classes=6, num_view=5, view_list=None, fea_out=200, fea_com=300, **kwargs):
    MultiviewNet = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'multiviewNet':
        MultiviewNet = MultiViewNet(num_classes=num_classes, num_view=num_view, view_list=view_list,
                                    fea_out=fea_out, fea_com=fea_com, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(MultiviewNet, init_type=init_type)

    if use_gpu:
        MultiviewNet.cuda()

    if pretrained:
        MultiviewNet.load_state_dict(model_root)

    return MultiviewNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class MultiViewNet(nn.Module):
    def __init__(self, num_classes, num_view, view_list, fea_out, fea_com):
        super(MultiViewNet, self).__init__()

        # list of the linear layer
        self.linear_specific = []
        for i in range(len(view_list)):
            self.add_module('linear_' + str(i), nn.Sequential(
                nn.Linear(view_list[i], 2 * fea_out).cuda(),
                nn.BatchNorm1d(2 * fea_out).cuda(),
                nn.ReLU(inplace=True).cuda(),
                nn.Dropout().cuda(),
                nn.Linear(2 * fea_out, fea_out).cuda(),
                nn.BatchNorm1d(fea_out).cuda(),
                nn.ReLU(inplace=True).cuda()
            )
                            )
        self.linear_specific = AttrProxy(self, 'linear_')

        self.linear_shared = []
        for i in range(len(view_list)):
            self.linear_shared.append(nn.Sequential(
                nn.Linear(view_list[i], 2 * fea_out).cuda(),
                nn.BatchNorm1d(2 * fea_out).cuda(),
                nn.ReLU(inplace=True).cuda(),
                nn.Dropout().cuda(),
                nn.Linear(2 * fea_out, fea_out).cuda(),
                nn.BatchNorm1d(fea_out).cuda(),
                nn.ReLU(inplace=True).cuda()
            ))
        self.linear_shared = nn.ModuleList(self.linear_shared)

        self.relation_out = RelationBlock_Out(fea_out)

        self.linear_in = nn.Sequential(
            nn.Linear(num_view * fea_out, 2 * fea_out),
            nn.BatchNorm1d(2 * fea_out),
            nn.ReLU(inplace=True),
			nn.Linear(2 * fea_out, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(inplace=True)
        )

        self.linear_out = nn.Sequential(
            nn.Linear(int(num_view * (num_view-1) * fea_out / 2), 2 * fea_out),
            nn.BatchNorm1d(2 * fea_out),
            nn.ReLU(inplace=True),
			nn.Linear(2 * fea_out, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(inplace=True)
        )

        self.classifier_out = nn.Sequential(
            nn.Linear(fea_out, fea_com),
            nn.BatchNorm1d(fea_com),
            nn.ReLU(inplace=True),
            nn.Dropout(),
			nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, input):

        # extract features of inputs
        Spec_list = []
        Shared_list = []

        for input_item, linear_item in zip(input, self.linear_specific):
            fea_temp = linear_item(input_item)
            Spec_list.append(fea_temp)

        for input_item, linear_item in zip(input, self.linear_shared):
            fea_temp = linear_item(input_item)
            Shared_list.append(fea_temp)

        Mutual_list = self.relation_out(Spec_list)  # interation information
        Comple_temp = self.linear_in(torch.cat(Spec_list, 1))
        Commun_list = []
        for i in range(len(Spec_list)):
            Commun_list.append(Comple_temp + Spec_list[i])

        Shared_temp = torch.sum(torch.stack(Shared_list), dim=0) / len(Shared_list)
        Correl_temp = self.linear_out(torch.cat(Mutual_list, 1))

        Commun_list.append(Correl_temp)
        Commun_list.append(Shared_temp)
        Fea_Relation_list = []
        for k in range(len(Commun_list)):
            Fea_Relation_list.append(self.classifier_out(Commun_list[k]))

        return Fea_Relation_list, Shared_list


class RelationBlock_Out(nn.Module):
    def __init__(self, fea_out):
        super(RelationBlock_Out, self).__init__()
        self.fea_out = fea_out
        self.linear_out = nn.Sequential(
            nn.Linear(fea_out * fea_out, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(inplace=True),
        )

    def cal_relation(self, input1, input2):

        input1 = input1.unsqueeze(2)
        input2 = input2.unsqueeze(1)
        outproduct = torch.bmm(input1, input2)

        return outproduct

    def forward(self, x):

        relation_view_list = []
        for i in range(len(x)-1):
            for j in range(i+1, len(x)):
                relation_temp = self.cal_relation(x[i], x[j])
                relation_temp = relation_temp.view(relation_temp.size(0), self.fea_out * self.fea_out)
                relation_temp = self.linear_out(relation_temp)
                relation_view_list.append(relation_temp)

        return relation_view_list
