import torch
import src.get_neighbour as get_neighbour

def get_batch_neigh(com_coor,
                    cell_matrix,
                    species,
                    period,
                    neigh_atoms,
                    batchsize,
                    cutoff,
                    device):
    '''
    这个函数的作用是返回当前进程的 atom_index_rank 和 shifts_rank 
    
    工作原理:
        将当前rank分配到的数据里的数据进一步分为几个子batch, 方便计算,时间搜找邻近原子的功能是在get_neighbour中
    该函数负责计算多个点(com_coor中的坐标)及其对应的邻近原子对,考虑了周期性边界条件pbc和指定的截断半径cutoff来确定哪些原子是彼此的邻居
    n_tot_point_rank: 当前GPU分配到的结构数
    maxnumatom: 最大原子数
    cartesian: 每个子batch的坐标数据
    # n_max_atom_structure_rank * neigh_atoms 表示每个结构中最大邻居对数
    
    '''
    
    
    if True:
        n_tot_point_rank = com_coor.shape[0] 
        n_max_atom_structure_rank = com_coor.shape[1]    
        
        # shifts = [结构数, 最大组合数, 3] atom_index = [2, 组合数 , 最大组合数]
        shifts = torch.empty(n_tot_point_rank,n_max_atom_structure_rank*neigh_atoms,3) # 于存储每个原子的位移向量。在处理周期性边界条件（PBC）时，如果两个原子是邻居，shifts 会记录它们的位移矢量   
        atom_index = torch.empty((2,n_tot_point_rank,n_max_atom_structure_rank*neigh_atoms),dtype=torch.long) #代表两个矩阵，可以在这两个矩阵之间建立联系
        
        tmp_batch=1
        max_neigh=0 # 用来跟踪所有子batch中邻居原子数的最大值，确保在整个过程中能够处理到最大数量的邻居
    
    '''获取当前rank''' 
    for idx_point in range(1,n_tot_point_rank+1):
        # 按需计算, rank 里的所有结构 首先通过batchsize_per_GPU 分为迭代器返回的不同批次数据
        # 每个batchsize_per_GPU中的结构 可以进一步细分, 如果结构构象相同, 则减少计算量
        # 具体就是先检查晶胞，再检查两个结构中每个原子元素的构成，最后检查PBC
        
        # 一般来说, 假设所有结构都相似的话(原子, 晶胞不变, 原子序号也不变,只改变具体的xyz标), 这样效率比较高
        if      idx_point < n_tot_point_rank and \
                (cell_matrix[idx_point-1] == cell_matrix[idx_point]).all() and \
                (species[idx_point-1]== species [idx_point]).all() and \
                (period[idx_point-1] == period[idx_point]).all and\
                tmp_batch < batchsize:
            tmp_batch += 1 
       
        else:
            # 处理batchsize_per_GPU积累的不同类构象(子batch)的数据
            cartesian = com_coor[idx_point-tmp_batch:idx_point].to(device) # 只选择对应的结构切片, 张量形状不变
            species_ = species[idx_point-tmp_batch:idx_point].to(device)
            cell = cell_matrix[idx_point-tmp_batch].to(device) # 由于归位了相同类似构象，所以只选择一个晶格常数就行
            pbc = period[idx_point-tmp_batch].to(device)       # 由于归位了相同类似构象，所以只选择一个周期性边界条件就可以了  
            
            # 每一个相似构象的子bathc内计算相邻原子
            tmp_index,tmp_shifts,neigh = get_neighbour.neighbor_pairs(pbc, cartesian, species_, cell, cutoff, neigh_atoms)
            
            # 将在子batch内得到的结果添加到当前rank中的最终返回值 atom_index_rank, shifts_rank 中
            atom_index[:,idx_point-tmp_batch:idx_point] = tmp_index.to("cpu")
            shifts[idx_point-tmp_batch:idx_point] = tmp_shifts.to("cpu")
            
            max_neigh = max(max_neigh,neigh)
            torch.cuda.empty_cache()
            tmp_batch = 1 # 重新变为1,便于下一个batch的处理
    
    return shifts[:,0:max_neigh],atom_index[:,:,0:max_neigh]
