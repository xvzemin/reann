import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
from src.MODEL import *

#============================最终的模型汇总, 计算能量===================================
class Property(torch.nn.Module):
    def __init__(self,density,nnmodlist):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmodlist[0]
        if len(nnmodlist) > 1:  # 如果拟合的是极化率的话需要额外添加两个model
            self.nnmod1=nnmodlist[1]
            self.nnmod2=nnmodlist[2]

    def forward(self,cartesian,num_atoms,species,atom_index,shifts,create_graph=True): # 数据迭代器返回六个数据, 这里不需要label, 会在损失函数调用
        cartesian.requires_grad=True
        species = species.view(-1)                                                 # 由二维变成一维 
        
        '''正式开始前向传播'''
        # ① Property类的forward
        # ② GetDensity类的forward获得电子密度
        density = self.density(cartesian,num_atoms,species,atom_index,shifts)    #  应该是返回一个二维张量, 形状: [dataloader返回的结构数, 每个原子的嵌入嵌入密度]
        output = self.nnmod(density,species).view(num_atoms.shape[0],-1)         #  从一维张量转化为二维张量, 形状: [dataloader返回的结构数, 每个原子的嵌入能量]
        varene = torch.sum(output,dim=1)                                         #  求得每个结构的预测能量, 沿着第二个维度求和, 第二个维度消失, 即列消失
       
        grad_outputs=torch.ones(num_atoms.shape[0],device=cartesian.device)
        force=-torch.autograd.grad(varene,
                                   cartesian,
                                   grad_outputs=grad_outputs,
                                   create_graph=create_graph,
                                   only_inputs=True,
                                   allow_unused=True)[0].view(num_atoms.shape[0],-1)
        return varene,force

