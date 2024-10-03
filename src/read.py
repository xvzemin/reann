import os
import gc
import torch
import numpy as np
import torch.distributed as dist
from src.read_data import *
from src.gpu_sel import *
from src.get_info_of_rank import *
from datetime import datetime
from src.set_default_variable import set_default_variable
from src.xuzemin import xzm


'''创建训练情况的输出文件 nn.err 和 xzm自定义调试文件'''
if True:
    f_xzm = xzm('xzm.txt')                  # xuzemin自定义调试文件
    f_out = open('nn.err','w')              # 模型训练情况的输出文件nn.err
    f_out.write("REANN Package 开始运行\n"); start_time = datetime.now()

'''默认变量的设置, 输入文件input_nn和density的读取, 一些其他变量和模型结构的初步设置'''
if True:
    '''设置input_nn和input_density的默认变量'''
    if True:
        start_table = 0                # fit with force(1) or without force (0) DM(2), TDM(3), polarizability(4)
        table_coor = 0                 # 实际坐标，笛卡尔坐标(0), 分数坐标(1)
        table_init = 0                 # 新的训练(0), 从REANN.pth恢复训练(1)
        nblock = 1                      # >1 时表示引入残差神经网络, =1即普通的全连接层
        ratio = 0.9                     # 划分训练集和测试集的比例
        Epoch = 11                      # 最大训练批次
        patience_epoch = 100            # 训练n次后Loss没下降的调整
        decay_factor = 0.5              # 学习率的衰减因子
        print_epoch = 1                 # 每多少批次打印Loss
        start_lr = 1e-3                 # 初始学习率
        end_lr = 1e-5                   # 终止学习率
        re_ceff = 0.0                   # L2正则化系数
        batchsize_train = 32            # batch_size
        batchsize_val = 256             # batch_size
        e_ceff = 0.1                    # 能量的权重
        init_f = 10                     # 误差函数force权重初始值
        final_f = 0.5                   # 误差函数force权重终值
        nl = [128, 128]                 # 每层NN的节点数和深度
        dropout_p = [0.0, 0.0]          # 每层dropout的概率
        activate = 'Relu_like'          # 激活函数
        queue_size = 10                 # 线程预加载batch的数量
        table_norm = True               # 是否使用层归一化
        oc_loop = 1                     # 描述符的迭代次数
        oc_nl = [128, 128]              # 同nl，神经网络的宽度和深度
        oc_nblock = 1                   # 同nblock，残差神经网络
        oc_dropout_p = [0.0, 0.0]       # 同dropout_p
        oc_activate = 'Relu_like'       # 默认激活函数
        oc_table_norm = True             # 是否层归一化
        DDP_backend = "nccl"             # 分布式并行的后端
        folder = "./"                    # 数据目录
        dtype = 'float32'                # 数据类型
        norbit = None                    # 如果不指定norbit, 则默认初始化为(nwave+1)*nwave/2*(L+1)
        find_unused = False              
        nipsin = 2
        cutoff = 5.0
        nwave = 8
        patience_epoch = patience_epoch / print_epoch

    '''读取用户设置的 input_density, nipsin=L, +1是为了计算norbit, num_of_element为元素种类'''
    if True:
        with open('para/input_density','r') as f:
            while True:
                tmp = f.readline()
                if not tmp: break
                string = tmp.strip()
                if len(string) != 0:
                    if string[0] == '#':
                        pass
                    else:
                        m = string.split('#')
                        exec(m[0])
        nipsin += 1 # +1只是方便后续计算, 角量子数仍然是原来的
        num_of_element = len(atomtype)        

    '''读取用户设置的 input__nn, 并对变量赋值, 根据读取到的信息做一些设置'''
    if True:
        with open('para/input_nn','r') as f:
            while True:
                tmp = f.readline()
                if not tmp: break
                string = tmp.strip()
                if len(string) != 0:
                    if string[0] == '#':
                        pass
                    else:
                        m = string.split('#')
                        exec(m[0])                  # 执行所有变量的赋值
        
        # 设置torch的默认浮点数类型, np无法设置全局,但可以创建时传入np_dtype
        if dtype == 'float64':
            torch_dtype = torch.float64
            np_dtype = np.float64
        else:
            torch_dtype = torch.float32
            np_dtype = np.float32
        torch.set_default_dtype(torch_dtype) 

        # 将dropout_p和oc_dropout_p由列表转化为np数组
        dropout_p = np.array(dropout_p,dtype=np_dtype)
        oc_dropout_p = np.array(oc_dropout_p,dtype=np_dtype)    

        # 原始GTO参数的初始化, rs是中心, inta是alpha, 分别决定GTO的宽度和中心, rs和inta的形状:[元素种类, nwave] 
        if 'rs' in locals().keys(): # 检查是否自定义了rs和inta, 没有自定义就随机生成指定范围的参数
            rs = torch.from_numpy(np.array(rs,dtype=np_dtype))        
            inta = torch.from_numpy(np.array(inta,dtype=np_dtype))
            nwave = rs.shape[1]
            f_xzm.write(f"原始GTO参数rs和inta是自定义的,初始化为:\n{rs}\n{inta}\n") 
        else:
            rs = torch.rand(num_of_element,nwave)*cutoff                # rs∈[0, cutoff)
            inta =- (torch.rand(num_of_element,nwave)+0.2)              # inta∈[-1.2, -0.2)
            f_xzm.write(f"原始GTO参数rs∈[0, cutoff)和inta[-1.2, -0.2)随机初始化为:\n{rs}\n{inta}\n")        
        
        # 设置最终输出的物理量的神经元数量
        if start_table <= 2: outputneuron=1
        elif start_table == 3: outputneuron=3
        elif start_table ==4: outputneuron=1

        #选根据input_nn择计算描述符和描述符后网络的激活函数'''
        if oc_activate == 'Tanh_like': from src.activate import Tanh_like as oc_actfun
        else: from src.activate import Relu_like as oc_actfun
        if activate == 'Tanh_like': from src.activate import Tanh_like as actfun
        else: from src.activate import Relu_like as actfun

        # 根据input_nn选择需要拟合的性质,即整体数据网络的模型
        if start_table == 0: from src.Property_energy import *
        elif start_table == 1: from src.Property_force import *
        elif start_table == 2: from src.Property_DM import *
        elif start_table == 3: from src.Property_TDM import *
        elif start_table == 4: from src.Property_POL import *

        # ？？？？？？？？？？？？？？？？？？？？norbit的计算不懂为什么具体是这个数量
        if not norbit: # 如果自己不指定orbit, 那么norbit自动设置为(nwave+1)*nwave/2*(L+1)
            norbit = int((nwave+1)*nwave/2*(nipsin))
        nl.insert(0,norbit)
        oc_nl.insert(0,norbit)
                
        
        # ????????????????????????????????????????以下为势能面相关'''
        PES_Lammps=None  # 第二个if语句不知道作用
        if start_table <= 1: 
            import pes.script_PES as PES_Normal
            if oc_loop == 0:
                import lammps.script_PES as PES_Lammps
            else:
                import lammps_REANN.script_PES as PES_Lammps
        elif start_table == 2:
            import dm.script_PES as PES_Normal
        elif start_table == 3:
            import tdm.script_PES as PES_Normal
        elif start_table == 4:
            import pol.script_PES as PES_Normal




'''
--------------------------------------------------------------------------------------------------------------------------------------------------




                                                    以下代码正式开始调用各类模块


                        

--------------------------------------------------------------------------------------------------------------------------------------------------
'''


'''使用read.py中的函数Read_data()读取训练集和测试集的所有数据, 顺变定义了两个用于求RMSE的变量'''
if True:

    folder_train = folder+"train/"
    folder_test = folder+"test/"
    folder_list = [folder_train,folder_test]

    # 传递参数的数字代表拟合属性的维度, nprob: 拟合属性的维度
    if start_table == 0 or start_table == 1:
        n_of_train_and_test,\
        atom,mass,num_atoms,\
        cell_matrix,\
        pbc_for_all_points,\
        coor,\
        property,\
        force = Read_data(folder_list,\
                          1,\
                          start_table=start_table) # 还需要传递start_table时因为要区分处理单独处理能量和同时处理能量和力
    elif start_table == 2 or start_table == 3:
        n_of_train_and_test,\
        atom,\
        mass,\
        num_atoms,cell_matrix,\
        pbc_for_all_points,\
        coor,dip,\
        force=Read_data(folder_list,3)
    else:
        n_of_train_and_test,atom,\
        mass,\
        num_atoms,\
        cell_matrix,\
        pbc_for_all_points,\
        coor,pol,\
        force=Read_data(folder_list,9)

    # 对于有多少种结构每个结构的原子数，由列表转化为np数组
    n_of_train_and_test = np.array(n_of_train_and_test,dtype=np.int64) # n_of_train_and_test=[训练集结构数量，测试集结构数量]
    num_atoms = np.array(num_atoms,dtype=np.int64)
    n_tot_point = 0 ; n_tot_point = sum(n_of_train_and_test) # 计算train和test的总结构数存储于 n_tot_point--标量

    # 如果只有训练集train，没有测试集test，则按照input__nn中的ratio重新划分数据
    if n_of_train_and_test[1]==0: 
        n_of_train_and_test[0]=int(n_tot_point*ratio)
        n_of_train_and_test[1]=n_tot_point-n_of_train_and_test[0]

    # 所有结构的所有原子数量求和*3, 求力的RMSE时要用, 作为分母
    num_of_all_componnet_train = sum(num_atoms[idx_point] * 3 for idx_point in range(n_of_train_and_test[0]))
    num_of_all_componnet_test = sum(num_atoms[idx_point] * 3 for idx_point in range(n_of_train_and_test[0], n_tot_point))
                       
'''启动分布式训练, GPU并行处理, 分配给每个GPU自己的数据, 并对数据做一些加工方便之后直接传递给dataloader ''' 
if True:
    '''启动分布式训练, 调用gpu_sel函数, 选择显存最空的GPU, 有GPU就启动分布式训练, 没有GPU就用CPU''' 
    if True:
        local_rank = int(os.environ.get("LOCAL_RANK"))                      # 当前进程对应的GPU名
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE"))                # 当前节点的GPU数量
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >gpu_info')
        gpu_sel(local_size)                                                 # 根据当前节点的GPU数量尽量选择显存最空的GPU
        world_size = int(os.environ.get("WORLD_SIZE"))                      # 获取所有节点的总GPU数量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu",local_rank) # 如果是分布式计算必须要保证每个节点的GPU数量最好一致，否则速率会比较慢
        DDP_backend ="gloo"                                                  # 默认CPU协议，有GPU就用GPU
        if torch.cuda.is_available():
            DDP_backend = "nccl"
        dist.init_process_group(backend=DDP_backend)

    '''给每个GPU向下取整分配 batchsize_per_GPU, 不构分配剩下的个别数据舍弃 '''
    if True:
        # 下面两个主要是数据集不够用来提示报错的
        if batchsize_train < world_size or batchsize_val < world_size: # 防止每批次的数据样本数< GPU数量
            raise RuntimeError("batchsize太小比GPU数量小, 无法启动分布式训练")

        # min_data_len_train是传递给dataloader的数据长度，因此，有一些无法被整除的个别数据点会被舍弃
        batchsize_train = int(batchsize_train/world_size)
        batchsize_val = int(batchsize_val/world_size)
        n_point_per_gpu_upper_train = int(np.ceil(n_of_train_and_test[0]/world_size))
        n_point_per_gpu_upper_test = int(np.ceil(n_of_train_and_test[1]/world_size))
        n_point_per_gpu_lower_train = n_of_train_and_test[0]-n_point_per_gpu_upper_train*(world_size-1)
        n_point_per_gpu_lower = n_of_train_and_test[1]-n_point_per_gpu_upper_test*(world_size-1)
        if n_point_per_gpu_lower_train <= 0 or n_point_per_gpu_lower <= 0:
            raise RuntimeError("总结构数量太少，无法在这种情况下分布式训练")

    '''获取每个rank自己用的数据, 处理好的数据之后会传递给dataloader''' 
    if True:    
        # 判断当前是第几个进程(第几个GPU)
        rank=dist.get_rank()
        '''获取每个进程的分配到的结构索引范围 range_train 和 range_test'''
        if True: # 训练集 和 测试集
            rank_begin = n_point_per_gpu_upper_train*rank 
            rank_end = min(n_point_per_gpu_upper_train*(rank+1),n_of_train_and_test[0]) 
            range_train = [rank_begin,rank_end]
            
            rank_begin = int(np.ceil(n_of_train_and_test[1]/world_size))*rank
            rank_end = min(int(np.ceil(n_of_train_and_test[1]/world_size))*(rank+1),n_of_train_and_test[1])
            range_test = [n_of_train_and_test[0]+rank_begin,n_of_train_and_test[0]+rank_end]
        
        '''应用得到的每个进程的数据索引范围, 对自己进程的数据进行一些加工, 如坐标转化为相对于第一个原子和质心等'''    
        if True: # 生成传递给dataloader的6种数据, 前四种只是切片, 但符合距离条件的索引对atom_index和偏移向量shifts的求取比较复杂
            # 训练集
            com_coor_train_rank,\
            force_train,\
            numatoms_train,\
            species_train,\
            atom_index_train,\
            shifts_train=get_info_of_rank(range_train,
                                          atom,
                                          atomtype,
                                          mass,
                                          num_atoms,
                                          cell_matrix,
                                          pbc_for_all_points,
                                          coor,force,
                                          start_table,
                                          table_coor,
                                          neigh_atoms,
                                          batchsize_train,
                                          cutoff,device,
                                          np_dtype)
            # 测试集
            com_coor_val_rank,\
            force_val,\
            numatoms_val,\
            species_val,\
            atom_index_val,\
            shifts_val=get_info_of_rank(range_test,
                                        atom,
                                        atomtype,
                                        mass,
                                        num_atoms,
                                        cell_matrix,
                                        pbc_for_all_points,
                                        coor,force,
                                        start_table,
                                        table_coor,
                                        neigh_atoms,
                                        batchsize_val,
                                        cutoff,device,
                                        np_dtype)


# ????拟合其他性质这部分的切片没看
'''将需要拟合的物理性质的label切片为每个进程的label''' 
if True:
    n_prop = 1
    if start_table == 0: 
        property_train=torch.from_numpy(np.array(property[range_train[0]:range_train[1]],dtype=np_dtype))
        property_test=torch.from_numpy(np.array(property[range_test[0]:range_test[1]],dtype=np_dtype))
        property_train_rank=(property_train.view(-1),)
        property_test_rank=(property_test.view(-1),)
        
        num_of_point_and_triple_force_test=torch.empty(n_prop)
        num_of_point_and_triple_force_train=torch.empty(n_prop)
        num_of_point_and_triple_force_train[0]=n_of_train_and_test[0] 
        num_of_point_and_triple_force_test[0]=n_of_train_and_test[1] 

    if start_table == 1: # 给当前进程的ab_property具体分配数据，并转化为张量
        property_train=torch.from_numpy(np.array(property[range_train[0]:range_train[1]],dtype=np_dtype))
        property_test=torch.from_numpy(np.array(property[range_test[0]:range_test[1]],dtype=np_dtype))
        property_train_rank=(property_train.view(-1),force_train) # force_train 是重排后的力矩阵
        property_test_rank=(property_test.view(-1),force_val)
        n_prop=2
          
        # 计算RMSE的分母
        num_of_point_and_triple_force_train=torch.empty(n_prop)
        num_of_point_and_triple_force_test=torch.empty(n_prop)
        num_of_point_and_triple_force_train[0] = n_of_train_and_test[0]                  # 第一个维度为总训练集数量
        num_of_point_and_triple_force_train[1] = num_of_all_componnet_train              # 第二个维度为所有结构的所有原子数量的总和的3倍        
        num_of_point_and_triple_force_test[0] = n_of_train_and_test[1] 
        num_of_point_and_triple_force_test[1] = num_of_all_componnet_test

    if start_table == 2 or start_table == 3: 
        dip_train=torch.from_numpy(np.array(dip[range_train[0]:range_train[1]],dtype=np_dtype))
        dip_val=torch.from_numpy(np.array(dip[range_test[0]:range_test[1]],dtype=np_dtype))
        property_train_rank=(dip_train,)
        property_test_rank=(dip_val,)
        num_of_point_and_triple_force_test=torch.empty(n_prop)
        num_of_point_and_triple_force_train=torch.empty(n_prop)
        num_of_point_and_triple_force_train[0]=n_of_train_and_test[0]*3
        num_of_point_and_triple_force_test[0]=n_of_train_and_test[1]*3

    if start_table == 4: 
        pol_train=torch.from_numpy(np.array(pol[range_train[0]:range_train[1]],dtype=np_dtype))
        pol_val=torch.from_numpy(np.array(pol[range_test[0]:range_test[1]],dtype=np_dtype))
        property_train_rank=(pol_train,)
        property_test_rank=(pol_val,)
        num_of_point_and_triple_force_test=torch.empty(n_prop)
        num_of_point_and_triple_force_train=torch.empty(n_prop)
        num_of_point_and_triple_force_train[0]=n_of_train_and_test[0]*9
        num_of_point_and_triple_force_test[0]=n_of_train_and_test[1]*9 

 
'''init_pot和prop_ceff的设置 ''' 
if True:

    #  拟合能量的话 init_pot初始设置为每个原子的平均势能, 作为生成嵌入能量的初始偏置项, 不拟合能量的话就默认设置为0
    init_pot = 0.0
    if start_table <= 1:
        property = np.array(property,dtype=np.float64).reshape(-1)
        init_pot = np.sum(property)/np.sum(num_atoms)
    init_pot = torch.tensor([init_pot]).to(device).to(torch_dtype)

    # 设置能量和力的权重, 存储于prop_ceff, 并且转换dropout_为np数组''' 
    prop_ceff = torch.ones(2,device=device)
    prop_ceff[0] = e_ceff
    prop_ceff[1] = init_f

   
''' 删除不用的变量，释放显存''' 
if True:
    del coor,mass,num_atoms,atom,cell_matrix,pbc_for_all_points
    if start_table == 0: del property
    if start_table == 1: del property,force
    if start_table == 2 and start_table == 3: del dip
    if start_table == 4: del pol
    gc.collect()
 




