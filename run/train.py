#! /usr/bin/env python3
from src.read import *   # 训练过程的剩余代码在这个read.py文件中执行，其余模块都只定义了函数与类
import time
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dataloader import *
from src.optimize import *
from src.density import *
from src.MODEL import *
from src.EMA import *
from src.restart import *
from src.cpu_gpu import *
from src.Loss import *


# dataloader, 此时已经启动分布式训练了, 每个GPU有自己的dataloader
'''生成训练和测试数据迭代器, 有GPU的话还可以根据input_nn的queue_size选择是否提前加载一些数据'''
if True:
    dataloader_of_train=DataLoader(com_coor_train_rank,
                                   property_train_rank,
                                   numatoms_train,
                                   species_train,
                                   atom_index_train,
                                   shifts_train,
                                   batchsize_train,
                                   n_point_per_gpu_lower=n_point_per_gpu_lower_train,
                                   shuffle=True)

    dataloader_of_test=DataLoader(com_coor_val_rank,
                                  property_test_rank,
                                  numatoms_val,
                                  species_val,
                                  atom_index_val,
                                  shifts_val,
                                  batchsize_val,
                                  n_point_per_gpu_lower=n_point_per_gpu_lower,
                                  shuffle=False)

    if torch.cuda.is_available(): 
        dataloader_of_train=CudaDataLoader(dataloader_of_train,device,queue_size=queue_size)
        dataloader_of_test=CudaDataLoader(dataloader_of_test,device,queue_size=queue_size)
    else:
        dataloader_of_train = dataloader_of_train
        dataloader_of_test = dataloader_of_test
    
    ''' xzm自定义输出文件读取结束时间''' 
    if dist.get_rank() == 0:
        end_time = datetime.now()
        hours, remainder = divmod((end_time - start_time).total_seconds(), 3600); minutes, seconds = divmod(remainder, 60)
        f_out.write(f"数据加载花费的时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
        
'''   
 # 打印所有数据的形状并输出到xuzemi.txt文件
rank = dist.get_rank()
with open("xuzemi.txt", "w") as f:
    f.write(f"当前是 rank{rank}")
    f.write(f"坐标形状: {dataloader_of_train.coordinates_rank.shape}\n")
    f.write(f"标签: {[lbl.shape for lbl in dataloader_of_train.label]}\n")
    f.write(f"每个结构的原子数: {dataloader_of_train.num_atoms.shape}\n")
    f.write(f"元素种类: {dataloader_of_train.species_in_function.shape}\n")
    f.write(f"原子配对索引: {dataloader_of_train.atom_index.shape}\n")
    f.write(f"位移索引: {dataloader_of_train.shifts.shape}\n")
    f.write(f"Atom Index shape: {dataloader_of_train.atom_index}\n")
    f.write(f"Shifts shape: {dataloader_of_train.shifts}\n")
    f.flush()
''' 

'''模型的生成'''   
if True:
    # The first part of 模型
    '''计算描述符的模型'''
    if True:
        ocmod_list=[]       
        for ioc_loop in range(oc_loop):             # 描述符迭代的次数
            ocmod=NNMod(num_of_element,
                        nwave,
                        atomtype,
                        oc_nblock,
                        list(oc_nl),
                        oc_dropout_p,
                        oc_actfun,
                        table_norm=oc_table_norm)
            ocmod_list.append(ocmod)
        # 获得嵌入密度(包含了计算描述符的模型)
        getdensity=GetDensity(rs,inta,
                              cutoff,
                              neigh_atoms,
                              nipsin,
                              norbit,
                              ocmod_list,
                              f_xzm)

    # The second part of 模型
    '''描述符后的模型(嵌入密度到嵌入能量)'''
    if True:
        nnmod=NNMod(num_of_element,
                    outputneuron,
                    atomtype,
                    nblock,
                    list(nl),
                    dropout_p,
                    actfun,
                    init_pot=init_pot,
                    table_norm=table_norm)
        nnmodlist=[nnmod]

    # # The third part of 模型 (拟合极化率的时候才会有)
    '''拟合极化率的模型会多一部分模型, 拟合其他物理属性的pass这部分'''
    if start_table == 4:
        nnmod1=NNMod(num_of_element,
                     outputneuron,
                     atomtype,
                     nblock,
                     list(nl),
                     dropout_p,
                     actfun,
                     table_norm=table_norm)
        nnmod2=NNMod(num_of_element,
                     outputneuron,
                     atomtype,
                     nblock,
                     list(nl),
                     dropout_p,
                     actfun,
                     table_norm=table_norm)
        nnmodlist.append(nnmod1)
        nnmodlist.append(nnmod2)

    # 完整的模型, 并将模型传入DDP, 实现梯度同步
    '''合并两个部分, 并传入对应的Property类, 正式创建完整的模型'''
    if True:    
        Prop_class=Property(getdensity,nnmodlist).to(device).to(torch_dtype)
        # local_rank在read.py中定义，为当前GPU的进程
        if world_size > 1:
            if torch.cuda.is_available():
                Prop_class = DDP(Prop_class, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
            else:
                Prop_class = DDP(Prop_class, find_unused_parameters=find_unused)

# 损失函数
if True:
    loss_fn=Loss()

# 化器和学习率调度器
'''AdamW和ReduceLROnPlateau'''
if True:
    # 优化器，weight_decay=re_ceff是L2正则化系数，AdamW相比Adam更适合L2正则化
    optim=torch.optim.AdamW(Prop_class.parameters(), lr=start_lr, weight_decay=re_ceff)
    # 学习率调整器-Reduce Learning Rate on Plateau, factor=decay_factor为学习率衰减系数
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=decay_factor,
        patience=patience_epoch,
        min_lr=end_lr, 
        mode='min')

# 从REANN.pth继续训练的相关设置
if True:
    restart=Restart(optim)
    # 根据table_init从REANN.pth中决定是否继续训练
    if table_init == 1:
        restart(Prop_class,"REANN.pth")
        nnmod.init_pot[0]=init_pot # 这行代码不知道干啥的
        # 为什么restart的时候需要重新设置学习率，模型里没有么
        if optim.param_groups[0]["lr"]>start_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
        if optim.param_groups[0]["lr"]<end_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
        lr = optim.param_groups[0]["lr"]
        f_ceff = init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
        prop_ceff[1]=f_ceff

# Exponential Moving Average(EAM), 指数移动平均
'''平滑系数, 接近1表示对历史数据的重视程度很高, 也就是说每次更新的影响会比较小, 保留之前更多的权重信息'''
if True:
    ema = EMA(Prop_class, 0.999) # EMA (Exponential Moving Average)

'''
--------------------------------------------------------------------------------------


                                        以下正式开始训练


--------------------------------------------------------------------------------------
'''

if dist.get_rank()==0: # 计时用
    f_out.write(time.strftime("start: %Y年%m月%d日  %H时%M分%S秒 \n", time.localtime()))
    f_out.flush() # 强制将缓冲区中的数据写入到文件
    for name, m in Prop_class.named_parameters():
        print(name)

# 开始训练
Optimize(f_xzm,
         f_out,                                    # nn.err, 只有在主进程会读写文件
         prop_ceff,                                # 一维张量, 训练时动态调整力的系数的列表, 存储了loss加权计算的系数, 除了拟合拟合能量和力元素数量为2, 其余都为1 
         n_prop,                                   # 存储了要拟合性质的数量 1 或 2
         num_of_point_and_triple_force_train,      # 用loss计算RMSE用, 这个是分母， 列表, 第一个元素是每个GPU分配到的结构数量(每个GPU都相等), 第二个元素是所有结构所有原子*3,代表了有多少力的分量，方便后面计算RMSE 
         num_of_point_and_triple_force_test,       # 同上
         init_f,                                   # 初始在loss的力权重                                    
         final_f,                                  # 终止在loss的力权重  
         decay_factor,                             # 学习率衰减因子
         start_lr,                                 
         end_lr,                                   
         print_epoch,                              # print_epoch和模型的输出评率有关
         Epoch,                                
         dataloader_of_train,
         dataloader_of_test,
         Prop_class,                               # 总的模型
         loss_fn,                                  # 损失函数是简单的MSE, 但后续会对不同物理量的MSE做加权生成最终的loss
         optim,                                    # 优化器   
         scheduler,                                # 学习率调度器
         ema,                                      # EMA的实例                                   
         restart,                                  # 梯度爆炸时, 重启调整学习率, 继续优化
         PES_Normal,
         device, 
         PES_Lammps=PES_Lammps)

if dist.get_rank()==0: # 计时用
    end_time_main = datetime.now()    
    hours, remainder = divmod((end_time_main - start_time).total_seconds(), 3600) # # 计算小时、分钟和秒（包括小数部分）
    minutes, seconds = divmod(remainder, 60)
    f_out.write(time.strftime("end: %Y年%m月%d日  %H时%M分%S秒 \n", time.localtime()))
    f_out.write(f"程序运行总时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
    f_out.write("REANN Package 运行正常结束\n")
    f_out.close()