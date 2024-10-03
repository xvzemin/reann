import torch
import numpy as np
from src.read_data import * 
from src.get_batchneigh import *    
from src.com import *               # 坐标变换为相对于第一个原子与质心的计算模块

# 输入参数包括：点数、原子、质量、每个分子的原子数、晶格常数、周期性表、坐标、附加属性、力等信息
def get_info_of_rank(range_rank, 
                     atom, 
                     atomtype, 
                     mass, 
                     num_atoms, 
                     cell_matrix, 
                     pbc_for_all_points, 
                     coor, force,
                     start_table, 
                     table_coor, 
                     neigh_atoms, 
                     batchsize, 
                     cutoff, 
                     device, 
                     np_dtype):
    '''
    该函数的作用是通过每个进程的索引--range_rank--切片, 将通过Read_data读取到的数据分配给每个进程
    
    参数：
    - range_rank: 每个GPU分配的数据的起始索引和结束索引
    - atom: 原子的种类数组
    - atomtype: 原子类型的列表
    - mass: 每个原子的质量
    - num_atoms: 每个分子的原子数
    - cell_matrix: 晶格常数的矩阵
    - pbc_for_all_points: 周期性边界条件的表
    - coor: 原子的坐标
    - force: 力矩阵
    - start_table: 标志位，是否需要重排力矩阵
    - table_coor: 标志位，是否输入笛卡尔坐标
    - neigh_atoms: 邻居原子相关信息
    - batchsize: 每个批次的大小
    - cutoff: 截断距离
    - device: 设备(CPU或GPU)
    - np_dtype: numpy数据类型
    - max_num_atom_rank: 标量
    
    返回值：
    - com_coor_rank: 相对质心的坐标
    - reordered_force_rank: 重排后的力矩阵
    - num_atoms_rank: 每个分子的原子数
    - species_rank: 原子所属的元素类型索引
    - atom_index_rank: 原子的索引
    - shifts_rank: 位移信息
    '''
    
    
    
    
    ''' 获取num_atoms_rank, species_rank, 并调用get_com获取 com_coor_rank, reordered_force_rank '''
    if True:
        # 切片操作，获取每个进程对应的数据
        atom_rank = atom[range_rank[0]:range_rank[1]]           # 获取当前进程的原子数据
        mass_rank = mass[range_rank[0]:range_rank[1]]           # 获取当前进程的原子质量数据
        num_atoms_rank = num_atoms[range_rank[0]:range_rank[1]]  # 获取当前进程的原子数量数据
        
        max_num_atom_rank = max(num_atoms_rank)                 # 找出当前进程中最大的原子数
        
        # 获取当前进程的晶格常数、周期性边界条件、坐标、力
        cell_rank = np.array(cell_matrix[range_rank[0]:range_rank[1]], dtype=np_dtype)             # 获取当前进程的晶格常数
        pbc_rank = np.array(pbc_for_all_points[range_rank[0]:range_rank[1]], dtype=np.int64)       # 获取当前进程的周期性边界条件
        coor_rank = coor[range_rank[0]:range_rank[1]]                                              # 获取当前进程的坐标数据
        force_rank = None                                                                          # 初始化力矩阵
        if start_table == 1:                                                                     
            force_rank = force[range_rank[0]:range_rank[1]]                                        # 如果需要力,那么也对当前进程的力切片
        
        # 生成一个超大二维张量 species_rank，每一行是一个样本，元素依次是该原子所属元素在 atomtype 里的索引
        # 两层循环依次遍历每个结构，将species_rank对应的值替换为元素索引，如果该结构对应的位置不存在原子，则索引为-1
        species_rank = -torch.ones((range_rank[1]-range_rank[0], max_num_atom_rank), dtype=torch.long) 
        for idx_point in range(range_rank[1]-range_rank[0]):                                       # 遍历当前进程的所有点（结构）
            for idx_type, ele in enumerate(atomtype):                                              # 从 atomtype 中提取出每个结构对应原子所属元素在 atomtype 的索引
                mask = torch.tensor([ele_in_atom == ele for ele_in_atom in atom_rank[idx_point]])  # 掩码
                ele_index = torch.nonzero(mask).view(-1)                                           # nonzero方法可以提取元素为1的索引，由于范围的是二维张量，所以要重塑为1维
                if ele_index.shape[0] > 0:  # 如果找到该元素
                    species_rank[idx_point, ele_index] = idx_type                                  # 将每个结构中每个原子所属元素的索引赋值给 species_rank

        # 调用 get_com 函数，获取相对质心坐标和和力矩阵
        com_coor_rank, reordered_force_rank = get_com(coor_rank, force_rank, mass_rank, cell_rank, num_atoms_rank, max_num_atom_rank, table_coor, start_table)
        
        if start_table == 1:  
            reordered_force_rank = torch.from_numpy(reordered_force_rank)    # 将重排后的力矩阵转换为PyTorch张量
        com_coor_rank = torch.from_numpy(com_coor_rank)              # 将相对质心的坐标转换为PyTorch张量
        cell_rank = torch.from_numpy(cell_rank)                      # 将晶格常数转换为PyTorch张量
        num_atoms_rank = torch.from_numpy(num_atoms_rank)              # 将原子数转换为PyTorch张量   
        pbc_rank = torch.from_numpy(np.array(pbc_rank))              # 将周期性边界条件转换为PyTorch张量


    '''调用 get_batch_neigh 函数, 获取 atom_index_rank 和 shifts_rank'''
    if True:
        # 实际上 num_atoms * neigh_atoms 就是最大的可能的组合数
        # atom_index_rank=[2, 结构数, 组合数]   shifts_rank=[结构数, 组合数, 3]
        # atom_index = (2, num_mols, num_atoms * neigh_atoms)
        # shifts = -1e11 * torch.ones((num_mols, num_atoms * neigh_atoms, 3)
        shifts_rank, atom_index_rank = get_batch_neigh(com_coor_rank, 
                                                       cell_rank, 
                                                       species_rank, 
                                                       pbc_rank, 
                                                       neigh_atoms, 
                                                       batchsize, 
                                                       cutoff, 
                                                       device) 
    # 结合    get_info_of_rank 中的两个小模块, 返回每个进程的dataloader所需要的数据
    return com_coor_rank, reordered_force_rank, num_atoms_rank, species_rank, atom_index_rank, shifts_rank  

