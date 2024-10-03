import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np
import opt_einsum as oe   # 爱因斯坦求和约定部分的操作,  ① torch.einsum 适合中小张量  ② oe.contract  张量较大的时候效率更高
                                                      


class GetDensity(torch.nn.Module):
    def __init__(self,rs,inta,cutoff,neigh_atoms,nipsin,norbit,ocmod_list,f_xzm):
        super(GetDensity,self).__init__()
        '''
        '''
        # 参数初始化
        if True:
            self.f_xzm = f_xzm

            self.nipsin = nipsin                                                               # 角度函数的阶数
            # 4个可训练参数 rs, inta, contracted_orbit_coefficient, hyper
            self.rs = nn.parameter.Parameter(rs)                                               # 径向函数的参数, 依赖于元素
            self.inta = nn.parameter.Parameter(inta)                                           # 径向函数的指数系数, 依赖于元素
            self.params = nn.parameter.Parameter(torch.ones_like(self.rs)/float(neigh_atoms))  # 从基组组成收缩原子轨道的系数, 依赖于元素, 还有点类似让电子密度归一化的系数
            self.hyper = nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.rand(self.rs.shape[1],norbit)).\
            unsqueeze(0).repeat(nipsin,1,1))  # 每一种具体的L值对映一套自己独立的分子轨道系数
                                              # 生成[nipsin,nwave,norbit],每层数值相同 ，感觉这个就是存储的不同L下的inta和rs

            # 2个不可训练参数 cutoff和 index_para
            self.register_buffer('cutoff', torch.Tensor([cutoff]))                           # 截断距离,不可被训练的参数
            npara=[1]                                                                        # 每个角度展开阶数下的参数数量
            index_para=torch.tensor([0],dtype=torch.long)                                    # 记录每个参数所属的阶数
            for i in range(1,nipsin):                                                        # 逐步计算各阶的参数数量
                npara.append(np.power(3,i))                                                  # npara = [1, 3, 9]
                index_para=torch.cat((index_para,torch.ones((npara[i]),dtype=torch.long)*i)) # index_para = tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            self.register_buffer('index_para',index_para)                                    # 保存角度展开的参数索引


        # 从ocmod里合并为一个模型
        if True:
            ocmod = OrderedDict()                                                              
            for i, model in enumerate(ocmod_list):
                f_oc = "memssage_"+str(i)                                                      
                ocmod[f_oc] = model
            self.ocmod= torch.nn.ModuleDict(ocmod)                                           

    def gaussian(self,distances,species_): # [728, 7]
        '''指数部分的计算'''
        rs = self.rs.index_select(0,species_)                       # ① 同一元素使用相同的基组rs和inta, rs会变大很多, 行依次是每种结构每种组合元素所使用的rs和inta
        inta = self.inta.index_select(0,species_)                   # ② 一共有nwanve列, 每列分别代表了每种原始GTO的指数部分
        radial = torch.exp(inta*torch.square(distances[:,None]-rs)) # distances[:,None]由一维向量转化为二维张量, 再利用广播机制-rs ,之后按元素操作相乘
        return radial                                               # 返回的 radia 是一个二维张量 代表了每种结构每种组合的指数部分

    def cutoff_cosine_function(self,distances): # [728]
        '''截断函数部分的计算'''
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)  
                                                                    # 返回的截断函数f(r)是一个一维张量，相比指数部分
    
    def angular(self,dist_vector,cutoff_cosine): # [13, 728]
        '''xyz部分的计算, dist_vector是distance对应的二维矢量形式'''
        tot_neighbour = dist_vector.shape[0]                        # 小批次结构总的邻居数目, 也就是总的组合数
        dist_vector = dist_vector.permute(1,0).contiguous()         # 交换维度, n*3, 现在每列是向量, 3*n
        angular = [cutoff_cosine.view(1,-1)]                        # 将一维张量转化为二维行向量, angular是一个含有一个张量的列表
        for ipsin in range(1,int(self.nipsin)):                     # 逐步计算各阶次的角度展开
            #      [1,728] * [3,728] - [1,3,728]->[3,728]
            #       [3,728]*[3,728] - [3,3,728]->[9,728]
            angular.append(torch.einsum("ji,ki -> jki",angular[-1],dist_vector).reshape(-1,tot_neighbour))
            '''
                ① 先从 1*n 和 3*n 两个张量生成三维张量 1*3*n , 再reshape变成3*n 
                ② 计算得到的3*n这个张量的每一列分别是同一个组合内的f(r)*对应实际位移向量的三个分量,即xyz值
                ② 即每一列是( f(r)*x, f(r)*y, f(r)*z ) 重复n个组合
                ④ 一般情况下 nipsin = 3, 循环重复两次两次
                ⑤ angular这个列表最后含有三个元素, 即三个张量, 第一个张量是s 第二个是p, 第三个d
                    1. r 每一列( f(r), f(r), f(r) )
                    2. r 每一列( f(r)*x, f(r)*y, f(r)*z )
                    3. d 每一列( f(r)*x^2, f(r)*y^2, f(r)*z^2 ) 
                    4. f 每一列( f(r)*x^3, f(r)*y^3, f(r)*z^3 )
                ⑥ s部分因为L = 0, lx=ly=lz=0, 所以xyzf(r)部分就等于f(r)
                ⑦ p部分因为L = 1, lx,ly,lz三者任意为1, 所以xyzf(r)部分就等于f(r)*x或f(r)*y或f(r)*z
                ⑧ d部分因为L = 2, 其他部分同理
                ⑨ angular部分计算存储了所以的这些信息
                ⑩ 需要修正, 他这里认为xy和yx是不一样的, 所以会多一些, L=2时d不是6项而是9项
            ''' 
        return torch.vstack(angular)  # ① 将不同阶次的角度函数值拼接起来, 由列表转为张量, 形状为[nipsin*3,所有结构的组合数]
                                      # ② 也就是说只要访问每一列, 每一个组合,所需要的xyzf(r)部分的所有数据都可以得到

    def obtain_orb_coeff(self,tot_n_atom_dummy:int,orbital,atom_index12,orb_coeff,hyper):
        '''
        orbital是所有GTO的可能取值
        '''
        expand_para = orb_coeff.index_select(0,atom_index12[1])                              # [728, 7]所有原子的收缩原子轨道的系数 atom_index12[1]是所有可能的组合数
        worbital = oe.contract("ijk,ik->ijk", orbital,expand_para,backend="torch")           # [728, 13, 7]从GTO计算加权后真正的的收缩原子轨道
        sum_worbital = torch.zeros((tot_n_atom_dummy,orbital.shape[1],self.rs.shape[1]),dtype=orbital.dtype,device=orbital.device)  # [728, 13, 7]初始化加权轨道的和
        # 所有中心原子的不同角量子数的加权GTO
        sum_worbital = torch.index_add(sum_worbital,0,atom_index12[0],worbital)              # [56, 13, 7]atom_index12[0]只包含所有中心邻近原子, 728个穷举组合中只有56对有效,对每个原子的加权轨道进行求和
        # 也就是说实际上到这一为止sum_worbital已经计算出了以第一根筷子为中心的,其它原子的加权GTO,但加权GTO还没有真正组成原子轨道
        
        hyper_worbital = oe.contract("ijk,jkm -> ijm",sum_worbital,hyper,backend="torch")    # [56, 13, 84]计算超参数加权的轨道函数
        # hyper = [nipsin,7,norbit]，这里的hyper已经通过index_para变大了
        # [56, 13, 7] [13,7,norbit] -> [56, 13, 84]
                                                                                             # hyper [13, 7, 84]
        self.f_xzm.write(f'orbital: {orbital.size()}\n')
        self.f_xzm.write(f'expand_para: {expand_para.size()}\n')
        self.f_xzm.write(f'w_orbital: {worbital.size()}\n')
        self.f_xzm.write(f'sum_w_orbital: {sum_worbital.size()}\n')
        self.f_xzm.write(f'hyper: {hyper.size()}\n')
        self.f_xzm.write(f'hyper_w_orbital: {hyper_worbital.size()}\n')
        self.f_xzm.write(f'self.hyper: {self.hyper.size()}\n')
        self.f_xzm.write(f'square: {torch.sum(torch.square(hyper_worbital),dim=1).size()  }\n',True)

        return torch.sum(torch.square(hyper_worbital),dim=1)                                # [56, 84]返回计算得到的平方和结果，作为电子密度
        

    def forward(self,cartesian,num_atoms,species,atom_index,shifts):
        index_shift_of_atom = torch.arange(num_atoms.shape[0],device=cartesian.device)*cartesian.shape[1]#[4]
        '''
        ① num_atoms.shape[0]代表该批次的结构数量,第一步生成一个代表每个分子索引的一维张量 
        ② 接着用这个张量*最大原子数cartesian.shape[1]
        ③ 这样新的张量就变为了tensor(0, num_max_atom,2倍num_max_atom,3倍num_max_atom.....)
        ④ 这个张量意味着第一个结构的第一个原子的索引从0开始, 第二个结构的第一个原子的索引从num_max_atom开始....依此类推
        ⑤ 之后需要把所有批次结构中的原子转化为一维形式, 所以需要对原子索引做一个偏移
        '''
        self_mol_index = index_shift_of_atom.view(-1,1).expand(-1,atom_index.shape[2]).reshape(1,-1)    #[1,728]
        '''
        目的: 生成当前批次的全局原子索引
              由于同一批次中的不同结构要要求处理, 无法直接使用atom_index里对数据同时处理
              因此这一步的self_mol_index变量的目的是 对于每个结构中的原子对, 都给这个索引对加上一些偏移
              保证可以同时区分不同结构的原子索引对
              这一过程的目的是为每个原子的索引增加一个偏移量，使它们在全局的原子索引中唯一，避免不同批次中的原子索引重复。
        ① 先将上一步的tmp_index转化为列向量
        ② atom_index.shape[2]代表的是最大组合数n, expand将复制tmp_index转化为一个二维张量, 第二个维度是最大组合数
        ③ 最后再转化为为一个行向量, 即 前几个元素都是重复的第一个结构的第一个原子的索引，接着是相同的第二个结构的第一个原子的索引
        ④ 即(1, n)变为(1, n * k) 的一维张量, k是重复次数atom_index.shape[2]，生成分子索引  
        '''
        cart = cartesian.flatten(0,1)                 # 从三维展平为二维[56,3]
        tot_n_atom_dummy = cart.shape[0]              # 总的原子数量, 总的原子数量会包含一些虚拟的原子, 有些结构比较小, 原子数量少
        # 生成一个一维张量,按顺序代表每个每个结构的所有最大组合数个shift是否有效,组合数量, 生成有效的平移量掩码, 只有所有分量大于 -1e10 的向量才被认为是有效的
        padding_mask = torch.nonzero((shifts.view(-1,3)>-1e10).all(1)).view(-1) # [728]一维张量padding_mask包含所有有效的平移量的索引
        '''
        ① shifts.view(-1,3)首先转化为一个二维张量, 每一行都是一个组合,总行数为组合数*结构数
        ② >-1e10 用来过滤无效平移向量, 可能是由于误差什么的？
        ③ all(1)沿着第二个维度, 即从每一行从左到右, 依次判断是否所有分量都>-1e10, 都大于时, 这个向量才是有效的
        ④ 这个有效的向量用True表示, 即张量变化为         结构数*最大组合数,1 的二维张量, 即列向量
        ⑤ 将列向量转化为行向量, 来依次表示每个位移向量是否有效
        ⑥ 实际上每个行向量的位移向量按照每个结构依次分隔, 位移向量在每个结构中里也是按照原本的顺序排列的
        ⑦ 最终形状: 一维张量, 行向量:最大组合数*结构数,1
        
        '''     
 

        atom_index12 = (atom_index.view(2,-1)+self_mol_index)[:,padding_mask]       # ([2, 728])根据只提取有效的原子索引
        '''
        目的: 生成当前批次的全局原子索引
              由于同一批次中的不同结构要要求处理, 无法直接使用atom_index里对数据同时处理
              因此这一步的self_mol_index变量的目的是 对于每个结构中的原子对, 都给这个索引对加上一些偏移
              保证可以同时区分不同结构的原子索引对
              这一过程的目的是为每个原子的索引增加一个偏移量，使它们在全局的原子索引中唯一，避免不同批次中的原子索引重复。
        
        
        ① atom_index.view(2,-1)展开为形状为2, 最大组合数*结构数的二维张量 
        ② 相当于 两长条 两根平行的筷子 每条分别是每个结构的索引 两长条依次对映
        ③ 利用广播, 再加 self_mol_index 表示 每条都加上索引的偏移 
        ④ 这样得到了可以区分不同结构的原子有效索引对
        '''     
        # ①根据全局原子索引, 从cart(前面处理后变成了通过可以全局索引访问每一行的原子)
        # ②选择出相应的原子坐标，并将其重新组织成适当的形状，以便后续计算距离向量
        # ③ 2,-1,3  可以想象-1依次代表每个结构的原子 到这里已经完成了一批数据结构的合并
        selected_cart = cart.index_select(0, atom_index12.view(-1)).view(2, -1, 3)  # [2, 728, 3]根据索引提取邻居原子的坐标
        shift_values = shifts.view(-1,3).index_select(0,padding_mask)               # [728, 3]提取有效的对应的平移量        
        '''计算当前批次所有结构的截断半径内, 配对原子的距离向量'''
        # 都是二维张量, 结构数*组合数
        dist_vector = selected_cart[0] - selected_cart[1] + shift_values            # [728, 3]       
        # dist_vector是每行依次代表每个结构的原子对的的距离向量                
        # distances为所有距离的一维张量                   
        distances = torch.linalg.norm(dist_vector,dim=-1)                           # [728]计算邻居原子之间的距离
        species_ = species.index_select(0,atom_index12[1])                          # [728]传递给依赖元素种类的GTO的指数部分
        '''
        目的: 从 species 中提取出所有与原子成对的邻居原子的种类信息
            ① atom_index  相当于 两长条 两根平行的筷子 每根分别是每个结构的索引 两长条依次对映
            ② 可以将第一条看做中心原子, 而第二根看做邻近原子
            ③ 这样的话根据邻近原子的索引atom_index12[1], 就可以提取出周围邻近原子具体是哪一种元素
            ④ 根据具体的元素才可以采用不同的rs和inta计算GTO的指数部分
        '''        

        '''得到一个三维张量 ijk 第一维是所有原子, 每个二维张量里包含了所有的GTO的信息'''
        orbital = oe.contract("ji,ik -> ijk",\
                              self.angular(dist_vector,self.cutoff_cosine_function(distances)),\
                              self.gaussian(distances,species_),\
                              backend="torch")                                      # [728, 13, 7]
                                                                                    # #  [13, 728][728, 7]->[728, 13, 7]
        self.f_xzm.write(f'\
angular: {self.angular(dist_vector,self.cutoff_cosine_function(distances)).size()}\n\
高斯: {self.gaussian(distances,species_).size()}\n\
index_shift_of_atom: {index_shift_of_atom.size()}\n\
self_mol_index: {self_mol_index.size()}\n\
cart: {cart.size()}\n\
ot_n_atom_dummy: {tot_n_atom_dummy}\n\
padding_mask: {padding_mask.size()}\n\
atom_index12: {atom_index12.size()}\n\
selected_cart: {selected_cart.size()}\n\
shift_values: {shift_values.size()}\n\
dist_vector: {dist_vector.size()}\n\
distances: {distances.size()}\n\
species_: {species_.size()}\n\
orbital: {orbital.size()}\n\
')

        
        # self.rs.shape[1] = nwave
        # orb_coeff只包含三种元素的的收缩轨道系数-> 原子轨道
        orb_coeff = torch.empty((tot_n_atom_dummy,self.rs.shape[1]),dtype=cartesian.dtype,device=cartesian.device)    # [728,7]创建存储轨道系数的张量
        mask = (species > -0.5).view(-1)                                                                              # 生成掩码，筛选有效原子
        orb_coeff.masked_scatter_(mask.view(-1,1),self.params.index_select(0,species[torch.nonzero(mask).view(-1)]))  # 根据掩码填充轨道系数
        '''
        目的: 
            ① species[torch.nonzero(mask).view(-1)]---1D张量首先得到所有有效原子的元素索引
            ② 根据索引选择对应的GTO元素组合系数(从基组组成收缩原子轨道的系数)
            ③ 根据掩码, 替换有效原子的收缩轨道系数
            ④ 最后orb_coeff每一行依次代表每个结构的一个原子, 每列依次为每个原始GTO的系数, 这个收缩原子轨道的系数只依赖于元素 
        '''    
        hyper = self.hyper.index_select(0,self.index_para)                                                            # 根据角度索引提取超参数
        density = self.obtain_orb_coeff(tot_n_atom_dummy,orbital,atom_index12,orb_coeff,hyper)                        # 计算电子密度
        self.f_xzm.write(f'最后返回的density形状{density.size()}')
        for ioc_loop, (_, model) in enumerate(self.ocmod.items()):                                                    # 遍历修正模块
            orb_coeff = orb_coeff + model(density,species)  # 实际首先在更新的也就是元素依赖的收缩系数                                                          #  更新轨道系数
            density = self.obtain_orb_coeff(tot_n_atom_dummy,orbital,atom_index12,orb_coeff,hyper)                    # 重新计算电子密度
            self.f_xzm.write(f'最后返回的density形状{density.size()}')
        return density  # 返回计算得到的电子密度
 