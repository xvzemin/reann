def set_default_variable():
    # 输入变量初始化
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
    norbit = None                    # norbit的定义
    find_unused = False              # norbit的定义
    nipsin = 2
    cutoff = 5.0
    nwave = 8

    return (start_table, table_coor, table_init, nblock, ratio, Epoch, patience_epoch,
            decay_factor, print_epoch, start_lr, end_lr, re_ceff, batchsize_train,
            batchsize_val, e_ceff, init_f, final_f, nl, dropout_p, activate, 
            queue_size, table_norm, oc_loop, oc_nl, oc_nblock, oc_dropout_p, 
            oc_activate, oc_table_norm, DDP_backend, folder, dtype, norbit, 
            find_unused, nipsin, cutoff, nwave)

# 在主程序中调用
default_variables = set_default_variable()

# 可以解包变量
(start_table, table_coor, table_init, nblock, ratio, Epoch, patience_epoch,
 decay_factor, print_epoch, start_lr, end_lr, re_ceff, batchsize_train,
 batchsize_val, e_ceff, init_f, final_f, nl, dropout_p, activate, 
 queue_size, table_norm, oc_loop, oc_nl, oc_nblock, oc_dropout_p, 
 oc_activate, oc_table_norm, DDP_backend, folder, dtype, norbit, 
 find_unused, nipsin, cutoff, nwave) = default_variables


