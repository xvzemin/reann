import math
import torch

@torch.jit.script  # 使用JIT编译提升性能
def neighbor_pairs(pbc, 
                   coordinates, 
                   species, cell, 
                   cutoff: float, 
                   neigh_atoms: int):
    """
    ①该函数的作用是计算原子周围的邻近原子, 具体实现如下:
    e.g 当我需要考虑一个原子a周围的所有邻近原子时, 可以这样计算:
            ① 当有一个晶胞内的原子b在截断半径内时, 记向量原子a与原子b之间的向量ab的距离为|ab|, 
            ② 然而截断半径比较大时, 截断半径内的其他晶胞内可能还存在与原子b等价的原子b',我们还需要考虑|ab'|是否也在截断半径内
            ③ 这时并不需要再去直接计算a与b'的实际距离|ab'|与截断半径相比, 作为替代, 我们可以通过矢量合成, 即ab' = ab + bb'
            ④ ab是我们已知的, 而bb' 可能的数量就是后面的组合数, bb'的所有可能取值就是shifts的每一行,
            ⑤ 因此, 对于完整的一个结构内的所有原子而言, 只需要计算所有两两原子(包括自身)之间可能的组合方式, (类似向量ab)
            ⑥ 每一种原子之间的组合方式都对映着多种shifts, 通过这样的计算就能一次性得到每个结构中每个原子与多个晶胞内所有原子的距离
            ⑦ 这时只需要再从中筛选出对应的在截断半径内的原子即可, 并且需要提取出对应两两原子搭配的索引与对映的多种shifts
            ⑧ 以上是一个结构的完整处理, 只需要再加一个维度就可以实现批处理
        shifts的组合有多少种,
    ② 批处理的数量: 处理一个结构类似的tmp_batch, tmp_batch <= 每个GPU的batchsize
    ③ 由于处理的结构类似, 实际上笛卡尔积的部分就是在处理单一的一个结构, 但是得到的结构可以用到构象类似的所有tmp_batch中的结构
    ④ 索引组合可以用与当前tmp_batch中的所有结构
    """
    '''计算可能的实际偏移量shifts - shifts = [组合数量, 3] '''
    if True:
        padding_mask = (species == -1)                     # padding_mask为True的地方代表是dummy原子, 生成掩码以过滤无效原子，之前生成的species_rank中为-1的地方代表这个结构不存在这个原子
        num_point_tmp_batch = padding_mask.shape[0]        # batchsize的子batch的结构数量
        num_atoms_tmp_batch = padding_mask.shape[1]        # 每个结构的原子数, 这个数量>=实际的原子结构数量
        coordinates = coordinates.detach()                 # 分离计算图，避免梯度计算
        cell = cell.detach()

        # 计算每个维度上重复周期性单元格的次数，以覆盖截断距离
        # e.g ①如果截断半径相对比较大, 对于每个结构的原子考虑周围邻近原子时, 就需要考虑晶胞外的原子,
        #     ②每个原子都需要考虑周围很多晶胞内的原子, 而num_repeats_per_dimension在乘上实际的晶格常数后就是所需要考虑的阶段半径内的所有原子
        num_repeats_per_dimension = [pbc[i] * torch.ceil(cutoff / torch.max(torch.abs(cell[:, i]))).to(cell.device) for i in range(3)]
        
        # 生成三个维度上周期性需要重复的次数
        r1 = torch.arange(-num_repeats_per_dimension[0], num_repeats_per_dimension[0] + 1, device=cell.device)
        r2 = torch.arange(-num_repeats_per_dimension[1], num_repeats_per_dimension[1] + 1, device=cell.device)
        r3 = torch.arange(-num_repeats_per_dimension[2], num_repeats_per_dimension[2] + 1, device=cell.device)

        # 生成所有周期性偏移的组合, 并使用晶胞矩阵将偏移量转换为实际的空间坐标偏移
        shifts = torch.cartesian_prod(r1, r2, r3)           # 二维矩阵，笛卡尔积，返回一个新张量，每一行是一个组合，形状为[组合数量,3]
        shifts = torch.einsum("ij,jk -> ik", shifts, cell)  # shifts*cell # 形状为[组合数量,3] 每一行的3个元素代表了xyz方向的位移


    '''得到单个结构中所有可能的组合方式(原子索引搭配与偏移量)'''
    if True:
    # 获取周期性偏移量的组合数量, 每种偏移的索引, 原子索引
        num_shifts = shifts.shape[0]                                        # shifts的组合数量
        all_shifts = torch.arange(num_shifts, device=cell.device)           # 一维向量，shifts的索引
        all_atoms = torch.arange(num_atoms_tmp_batch, device=cell.device)   # 一维向量，由于处理的是类似的结构, 原子数相同, 会生成一个一维张量代表原子的索引
        
        # 计算所有可能的原子对和周期性偏移组合 prod 的每行分别是对应的组合搭配索引
        # prod转置前每一行是可能搭配的索引方式, 转置后每一行是所有的可能组合的数量
        prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t().contiguous()
        shift_index = prod[0] # ① 提取第一个维度的所有数值, 这就是每一种组合中shifts的索引
        p12_all = prod[1:]    # ② 这样对于每一种组合, 我们可以根shift_index的索引方位shifts中对应shifts向量的具体数值
                              # p12_all 2维张量,[2，组合数], 所有可能的原子搭配  
        shifts_all = shifts.index_select(0, shift_index) # [组合数, 实际位移], 生成一个储存所有可能的张量，每一行是实际位移

    '''计算距离并筛选出在截断距离内的原子对'''
    if True:
        # -1 这个维度代表 了索引所在的维度
        selected_coordinates = coordinates[:, p12_all.view(-1)].view(num_point_tmp_batch, 2, -1, 3) # 最终的矩阵代表了所有组合可能
        
        # selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...]就是对应所有组合可能的ab
        # shifts_all就是对应所有组合可能的bb'
        # distances 最终是所有可能的组合的距离 结构数量*组合数*距离
        distances = (selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shifts_all).norm(2, -1) #在最后一个维度上计算范数 
        
        # 注意padding_mask的形状是当前rank的所有结构的形状大小, 所以需要一些额外操作
        # 首先使用p12_all提取出所有原子对的掩码, 之后使用掩码过滤无效原子对，即原子不存在的即无效的距离, -1这个维度是组合数
        padding_mask = padding_mask[:, p12_all.view(-1)].view(num_point_tmp_batch, 2, -1).any(1)  # 检查第二个维度, 只要有一个维度有原子存在(值不为-1),就是True
        distances.masked_fill_(padding_mask, math.inf)  # 将掩码为True的位置, 即只要有原子不存在的地方, 就将该位置设为无穷大

        # 初始化用于存储原子对和偏移量的张量 num_atoms_tmp_batch * neigh_atoms代表了最大可能的组合数量
        atom_index = torch.zeros((2, num_point_tmp_batch, num_atoms_tmp_batch * neigh_atoms), device=cell.device, dtype=torch.long)
        shifts = -1e11 * torch.ones((num_point_tmp_batch, num_atoms_tmp_batch * neigh_atoms, 3), device=cell.device)
        maxneigh = 0  # 记录最大邻居数

    # 遍历每个分子，筛选符合距离条件的原子对
    for inum_mols in range(num_point_tmp_batch):
        # 筛选出距离在截断范围内的有效原子对索引
        # 提取出对应的组合数索引，重塑为一维张量
        pair_index = torch.nonzero(((distances[inum_mols] <= cutoff) * (distances[inum_mols] > 0.01))).reshape(-1)
        # 存储原子对的索引和对应的偏移量
        atom_index[:, inum_mols, 0:pair_index.shape[0]] = p12_all[:, pair_index] # 将符合条件的组合数索引传递给atom_index
        shifts[inum_mols, 0:pair_index.shape[0], :] = shifts_all.index_select(0, pair_index)
        maxneigh = max(maxneigh, pair_index.shape[0])  # 更新最大邻居数 # pair_index.shape是所有结构的最大组合数

    return atom_index, shifts, maxneigh  # 返回原子对索引、偏移量和最大邻居数







