import torch
from collections import OrderedDict
from torch import nn
from torch.nn import Linear, Dropout, BatchNorm1d, Sequential, LayerNorm
from torch.nn import Softplus, GELU, Tanh, SiLU
from torch.nn.init import xavier_uniform_, zeros_, constant_
import numpy as np

# 定义残差块（ResBlock）
class ResBlock(nn.Module):
    def __init__(self, nl, dropout_p, actfun, table_norm=True):
        super(ResBlock, self).__init__()
        nhid = len(nl) - 1  # 隐藏层数目，减去输入层和输出层
        sumdrop = np.sum(dropout_p)  # dropout 概率的总和
        modules = []  # 用来存储各层模块的列表
        
        # 循环构建隐藏层及其后的模块
        for i in range(1, nhid):  # 遍历每个隐藏层
            modules.append(actfun(nl[i-1], nl[i]))  # 添加激活函数层
            if table_norm: 
                modules.append(LayerNorm(nl[i]))  # 如果启用 LayerNorm，则在每层后添加
            if sumdrop >= 0.0001: 
                modules.append(Dropout(p=dropout_p[i-1]))  # 如果 dropout 总和大于某个阈值，则添加 Dropout 层

            # 创建线性层，连接第 i 层和第 i+1 层
            linear = Linear(nl[i], nl[i+1])
            if i == nhid - 1:  # 如果是最后一个隐藏层
                zeros_(linear.weight)  # 最后一层的权重初始化为零
            else:
                xavier_uniform_(linear.weight)  # 其他层使用 Xavier 初始化
            zeros_(linear.bias)  # 偏置初始化为零
            modules.append(linear)  # 添加线性层到模块列表中

        self.resblock = Sequential(*modules)  # 将所有模块串联为一个顺序模块

    def forward(self, x):
        # 残差连接，输出为计算后的值加上输入的原始值
        return self.resblock(x) + x


# 计算原子能量的神经网络模型
class NNMod(torch.nn.Module):
    def __init__(self, maxnumtype, outputneuron, atom_type, nblock, nl, dropout_p, actfun, init_pot=torch.zeros(1), table_norm=True):
        """
        maxnumtype: 最大原子种类数
        nl: 神经网络的层数结构
        outputneuron: 输出层神经元数量 # e.g 总能
        atom_type: 系统中的元素种类
        init_pot: 可以设置一个初始的原子势能，不指定的话默认偏置项为0
        """
        super(NNMod, self).__init__()
        
        # 变量初始化
        self.outputneuron = outputneuron
        elemental_nets = OrderedDict()  # 用来存储每个元素的神经网络，键为元素名，值为对应的神经网络
        sumdrop = np.sum(dropout_p)     # dropout 概率总和

        with torch.no_grad():           # 禁止梯度计算，以提高构建效率
            nl.append(nl[1])            # resblock要求输入和输出神经元数量相同，所以将第一隐藏层的大小也作为输出层
            nhid = len(nl) - 1          # 隐藏层数目

            # 遍历每种元素, 构建每种元素的神经网络
            for element in atom_type:
                modules = []  # 用于存储当前元素的神经网络模块
                linear = Linear(nl[0], nl[1])  # 输入层到第一隐藏层的线性映射
                xavier_uniform_(linear.weight)  # 使用 Xavier 初始化权重
                modules.append(linear)  # 添加线性层到模块中 # 激活函数接下来在resblock中
                
                # 添加多个残差块
                for iblock in range(nblock): # nblock默认为1，至少添加一次
                    modules.append(*[ResBlock(nl, dropout_p, actfun, table_norm=table_norm)])
                
                # 最后一个隐藏层的激活函数
                modules.append(actfun(nl[nhid-1], nl[nhid]))

                # 输出层，连接到最后的输出神经元
                linear = Linear(nl[nhid], self.outputneuron) # 最后一个隐藏层得到的数据经过激活函数才是各个嵌入能，嵌入能再全连接到总能
                zeros_(linear.weight)                        # 输出层的权重初始化为零
                linear.bias[:] = init_pot[:]                 # 偏置初始化为每个结构的平均原子势能
                modules.append(linear)                       # 添加输出层到模块中

                # 将构建好的神经网络模块存入字典，键为元素名
                elemental_nets[element] = Sequential(*modules) # 每个模型有对应的键和值，键为元素，值为对应的模型
        
        # 将所有元素的神经网络模块保存为一个 ModuleDict，便于后续调用
        self.elemental_nets = nn.ModuleDict(elemental_nets)

    def forward(self, density, species):
        """
        前向传播函数
            density: 应该是一维or二维张量, 形状: [dataloader返回的结构数, 每个原子的嵌入嵌入密度]
            species: 一维张量, 依次是每个结构每个原子的元素种类索引
        """
        # 创建一个全零的输出张量, 拟合能量的话形状就是一个二维列向量, 拟合其他性质的话就再添加几个列向量, 每一行代表结构数*最大原子数
        output = torch.zeros((density.shape[0], self.outputneuron), dtype=density.dtype, device=density.device)
        # 遍历每种元素，计算其对应的输出
        for idx_element_type, (_, element_model) in enumerate(self.elemental_nets.items()):
            mask = (species == idx_element_type)        # 获取当前所有结构, 对应元素原子的掩码
            ele_index = torch.nonzero(mask).view(-1)    # 从掩码获取当前元素类型在输入中的所有索引, 结构1的所有原子, 结构2的所有原子....
            
            if ele_index.shape[0] > 0:                      # ele_index: 一维张量, 依次是每个结构的所有该类型的元素索引
                ele_den = density[ele_index].contiguous()   # 利用这些原子索引访问对映的嵌入密度 
                output[ele_index] = element_model(ele_den)  # ① 这个模型的输出应该是最后的残差块的forward, 使用该元素对应的网络进行前向计算，并将结果写入输出张量中
                                                            # ② ele_den依次是每个结构对应元素原子的嵌入密度, 由每个嵌入密度根据对应的元素模型获得嵌入能量
        return output  # 返回一个一维张量, 依次是每个结构每个原子的嵌入能量

