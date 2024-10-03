import torch
import numpy as np
import torch.distributed as dist

class DataLoader():
    def __init__(self,
                 coordinates_rank,
                 label,                   # 拟合的物理属性
                 num_atoms,
                 species_in_function,
                 atom_index,
                 shifts,
                 batchsize,
                 n_point_per_gpu_lower=None,
                 shuffle=True):
        
        '''变量初始化'''
        if True:        
            self.coordinates_rank = coordinates_rank
            self.label = label
            self.species_in_function = species_in_function
            self.num_atoms = num_atoms
            self.atom_index = atom_index
            self.shifts = shifts
            self.batchsize = batchsize
            self.shuffle = shuffle
            self.end = self.coordinates_rank.shape[0]        # 结构数量
        
        '''shuffle是否打乱'''
        if self.shuffle:
            self.shuffle_list = torch.randperm(self.end)         # 生成一个(0~结构数-1)的随机一维张量
        else:
            self.shuffle_list = torch.arange(self.end)
        if not n_point_per_gpu_lower:
            self.min_data =self.end                      
        else:                                                    # 应该都是执行else, 前面如果分配到的结构数量为0的话会报错
            self.min_data = n_point_per_gpu_lower
        
        # self.length这个属性没用到, 这里相当与迭代器能够返回多少批次, 可能又会少几个除不尽的数据, self.batchsize = 设定的batchsize / 总GPU数
        self.length = int(np.ceil(self.min_data/self.batchsize)) 
        #print(dist.get_rank(),self.length,self.end)
      
    def __iter__(self):
        self.ipoint = 0                                          # 迭代器每给出一批结构, 计数在__next__自动+批次大小
        return self

    def __next__(self):
        if self.ipoint < self.min_data:
            index_batch = self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)] # 给出的结构的索引, 这一步看起来最后一个批次可能会处理不够一个批次的数据
            
            # index_select返回的还是原来形状的张量, 只不过只是特定的选择的结构
            coordinates = self.coordinates_rank.index_select(0,index_batch) 
            num_atoms = self.num_atoms.index_select(0,index_batch)    
            species = self.species_in_function.index_select(0,index_batch)
            shifts = self.shifts.index_select(0,index_batch)
            
            atom_index = self.atom_index[:,index_batch]     # 切片和index_select一样, 切完的形状和原来相同    
            
            abprop = (label.index_select(0,index_batch) for label in self.label) # 如果同时拟合能量和力, 返回两个张量, 其余返回对应的属性

            self.ipoint += self.batchsize
            #print(dist.get_rank(),self.ipoint,self.batchsize)
            return abprop,coordinates,num_atoms,species,atom_index,shifts
        else:
            # 打乱下一个epoch的数据
            if self.shuffle:
                self.shuffle_list=torch.randperm(self.end)
            raise StopIteration
