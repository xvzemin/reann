import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
# from torchviz import make_dot


def Optimize(f_xzm,
             f_out,
             prop_ceff,
             n_prop,                              # n_prop拟合的属性数量-普通标量
             num_of_point_and_triple_force_train,
             num_of_point_and_triple_force_test,
             init_f,
             final_f,
             decay_factor,
             start_lr,
             end_lr,
             print_epoch,
             Epoch,
             dataloader_of_train,
             dataloader_of_test,
             Prop_class,
             loss_fn,
             optim,
             scheduler,
             ema,
             restart,
             PES_Normal,
             device,
             PES_Lammps=None
             ): 

    rank = dist.get_rank()
    best_loss = 1e30*torch.ones(1,device=device) # 初始化best_loss为一个非常大的数, 确保可以更新损失, 跟踪到最低损失 
    
    for epoch in range(Epoch): 
        if rank == 0:
            f_xzm.write(f"--------------------Epoch:{epoch} 训练开始--------------------")
        

        '''模型训练部分'''
        if True:
            Prop_class.train()                                           # 将模型设置为训练模式，这会对Dropout和Batch Normalization有一些影响
            loss_prop = torch.zeros(n_prop,device=device)                # loss_prop是每个物理量的loss(MSE), loss是不同物理量加权后的loss                  
            for data in dataloader_of_train:                             # 这个循环执行完毕时，一个epoch已经训练完毕了，但还没正式使用EMA来调整参数
                abProp,cartesian,num_atoms,species,atom_index,shifts = data  # 迭代器返回的数据
                
                '''前向传播, 自动调用模型Prop_class的forward方法'''
                output = Prop_class(cartesian,num_atoms,species,atom_index,shifts)
                loss = loss_fn(output,abProp)                            # 计算损失值, 前向传播
                loss_prop += loss.detach()                               # 将损失累加, 这个最终存储的是当前epoch的[能量的loss, 力的loss]
                loss = torch.sum(torch.mul(loss,prop_ceff[0:n_prop]))    # 计算能量和力的loss的加权和
                optim.zero_grad(set_to_none=True)                        # 清楚梯度，防止积累,开True可以不给grad属性初始化为0张量，减少内存消耗               
                '''反向传播'''                                            # print(torch.cuda.memory_allocated)  # obtain the gradients  
                loss.backward()                                          # 反向传播时参数根据当前批次的数据更新, 不同GPU的模型参数已经自动同步
                optim.step()                                             # 计算梯度
                ema.update()      # ？？？？？？？？？？？？？？？记录当前下批量训练完的参数, 感觉可以优化成只记录最后一个数据的时候       # 计算指数平均后的参数，但还为将这个值加入模型
                '''
                # 生成计算图
                if i == 0 and rank == 0:
                    dot = make_dot(output, params=dict(Prop_class.named_parameters())) # 生成计算图
                    dot.render('model_structure', format='png')
                    # make_dot(output).render("autograd_graph", format="png")
                    
                    # 创建 TensorBoardX 的 SummaryWriter 实例
                    #writer = SummaryWriter(log_dir='model_graph')

                    # 使用 add_graph 来记录模型的计算图
                    #writer.add_graph(Prop_class, output)

                    # 关闭 writer，确保日志写入文件
                    #writer.close()
                    i += 1
                '''
        if rank == 0:
            f_xzm.write(f"--------------------Epoch:{epoch} 训练结束--------------------",  2)   

        
        '''模型预测部分'''
        # 每训练print_epoch次, 输出训练情况, 保存模型, 并调整学习率和力的权重
        if np.mod(epoch,print_epoch) == 0:                               # 求余数
            ema.apply_shadow(); Prop_class.eval()                        # 要评估了, 所以设置模型为平滑后的参数, 减小噪声, 并且开启评估模式                         
            
            '''计算训练的RMSE, 并输出到nn.err'''
            if True:
                dist.all_reduce(loss_prop,op=dist.ReduceOp.SUM)          # Reduction Operation是分布式计算的归约操作, 即将多个进程的loss_prop求和, 并且执行后，loss_prop的值在所有进程都是这个
                if rank == 0:
                    loss_prop = torch.sqrt(loss_prop.detach().cpu()/num_of_point_and_triple_force_train) # 按元素操作, 得到训练部分的RMSE
                    lr = optim.param_groups[0]["lr"]                     #  获取optim优化器中的第一个参数组中变量lr的值
                    f_out.write("{:<6} {:^4} {:^8} {:5e} {:8>} ".format("Epoch =", epoch, "学习率:", lr, "训练误差:"))
                    for error in loss_prop:
                        f_out.write('{:10.5f} '.format(error))
          
            '''得到测试时实际的loss值(loss_prop)和加权loss值'''
            if True:
                loss_prop = torch.zeros(n_prop,device=device)
                for data in dataloader_of_test:
                    abProp,cartesian,num_atoms,species,atom_index,shifts=data
                    loss = loss_fn(Prop_class(cartesian,num_atoms,species,atom_index,shifts,create_graph = False),abProp)
                    loss_prop = loss_prop + loss.detach()
                dist.all_reduce(loss_prop.detach(),op=dist.ReduceOp.SUM)
                loss = torch.sum(torch.mul(loss_prop,prop_ceff[0:n_prop]))   # 预测时的加权loss

            '''计算测试的RMSE, 并输出到nn.err'''
            if True:
                if rank == 0:
                    loss_prop = torch.sqrt(loss_prop.detach().cpu()/num_of_point_and_triple_force_test) # 计算测试的RMSE
                    f_out.write('{:8>}'.format("测试误差:"))
                    for error in loss_prop:
                        f_out.write('{:10.5f} '.format(error))
                    f_out.write("\n")
                    f_out.flush()

            '''根据测试的loss的值动态调整损失函数中力的加权值'''
            if True:
                scheduler.step(loss)                                # 根据每个print_epoch验证集的loss判断是否需要调整学习率
                lr = optim.param_groups[0]["lr"]                    # 获取第一个参数组中lr的值
                f_ceff = init_f + (final_f-init_f)*(lr-start_lr) / (end_lr-start_lr+1e-8) # 线性插值函数, 力在能量中的权重更具lr动态的从init_f调整到final_f
                prop_ceff[1] = f_ceff                                 # 每个print_epoch会根据学习率动态更新力的权重
            
                                   
            #  ？？？？？？？？？？？？？？？PES的相关设置每个print_epoch 选择性保存最优的模型参数 
            if loss < best_loss[0]:
                best_loss[0] = loss
                if rank == 0:
                    state = {'reannparam': Prop_class.state_dict(), 'optimizer': optim.state_dict()}
                    torch.save(state, "./REANN.pth")
                    PES_Normal.jit_pes()
                    if PES_Lammps:
                        PES_Lammps.jit_pes()
            
            '''预测结束, 将模型参数恢复为不是用EMA的参数'''
            ema.restore()
            
            # 梯度爆炸or出现异常时候, 调整学习率并重启模型 , 继续训练
            if loss > 25*best_loss[0] or loss.isnan():
                '''
                dist.barrier()确保所有进程都同步到异常处理：在分布式训练中，如果一个进程检测到了损失值异常
                （如 loss > 25 * best_loss[0] 或 loss.isnan()），它会进行模型恢复和学习率调整等操作。
                为了避免其他进程继续使用已经错误的参数进行训练, dist.barrier() 确保所有进程都在此同步，然后才会进行模型恢复。

                防止文件操作冲突: 例如, restart(Prop_class,"REANN.pth") 是在异常情况下从磁盘加载模型的操作，
                多个进程可能会同时尝试读取或写入相同的文件。dist.barrier() 可以避免多个进程在文件操作上产生冲突，
                确保只有在所有进程都到达同一位置时再执行文件恢复操作。
                '''
                dist.barrier()
                restart(Prop_class,"REANN.pth") # 重新加载模型并且不用学习率调度见，而是通过下面的代码手动更新学习率
                optim.param_groups[0]["lr"]=optim.param_groups[0]["lr"]*decay_factor
                ema.restart()
            
            '''要么epoch到最大设定, 要么学习率到指定降低值了才停止训练'''
            if lr <= end_lr: 
                # 一个epoch训练完毕                
                break         
 


