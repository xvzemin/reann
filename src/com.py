import numpy as np

# 中心质心（Center of Mass，COM）
# 该函数用于在考虑周期性边界条件（PBC）的情况下，将每个分子的坐标变换为相对于质心的坐标，以确保系统的平移不变性。
# 同时可以选择对力矩阵进行重排处理。

def get_com(coor, force, mass, scalmatrix, numatoms, maxnumatom, table_coor, start_table):
    ''' 
    该函数通过计算每个分子的质心，并将每个分子的坐标转换为相对于质心的坐标。
    同时可以对输入的力矩阵按照分子的原子顺序进行重新排列。
    
    参数：
    - coor: 分子坐标，形状为 [总点数, 最大原子数, 3] 的三维数组。
    - force: 每个原子所受的力，形状为 [总点数, 最大原子数, 3]。
    - mass: 每个原子的质量，形状为 [总点数, 最大原子数]。
    - scalmatrix: 尺度矩阵（晶胞矩阵），形状为 [总点数, 3, 3]。
    - numatoms: 每个分子的原子数，形状为 [总点数] 的一维数组。
    - maxnumatom: 每个分子的最大原子数，用于初始化相关数组。
    - table_coor: 标志位，若为0表示输入坐标为笛卡尔坐标，需转换为分数坐标。
    - start_table: 标志位，若为1则需要对力矩阵进行重排处理。

    返回：
    - com_coor: 相对质心转换后的坐标，形状为 [总点数, 最大原子数, 3] 的三维数组。
    - reordered_force: 如果需要，对力矩阵进行重排后的结果，形状为 [总点数, 最大原子数 * 3]。
    '''
    
    # 变量初始化部分
    ntotpoint = len(coor)  # 获取总的分子数（或总的坐标点数）
    maxnumforce = maxnumatom * 3  # 每个分子的最大力矩阵的大小，即每个原子有3个分量（x、y、z），所以是最大原子数的3倍
    reordered_force = None  # 初始化力矩阵的重排数组
    com_coor = np.zeros((ntotpoint, maxnumatom, 3), dtype=scalmatrix.dtype)  # 用于存储相对质心的分子坐标的三维数组
    fcoor = np.zeros((maxnumatom, 3), dtype=scalmatrix.dtype)  # 临时存储每个分子的坐标
    
    # 如果需要对力矩阵进行重排
    if start_table == 1:  
        reordered_force = np.zeros((ntotpoint, maxnumforce), dtype=scalmatrix.dtype)  # 初始化重排后的力矩阵
    
    # 遍历每个分子
    for ipoint in range(ntotpoint):  
        tmpmass = np.array(mass[ipoint], dtype=scalmatrix.dtype)  # 获取当前分子中每个原子的质量
        matrix = np.linalg.inv(scalmatrix[ipoint])  # 计算当前分子的尺度矩阵的逆矩阵

        fcoor[0:numatoms[ipoint]] = coor[ipoint]  # 将当前分子的坐标存储在临时变量 fcoor 中
        
        # 如果需要重排力矩阵
        if start_table == 1:  
            reordered_force[ipoint, 0:numatoms[ipoint]*3] = np.array(force[ipoint], dtype=scalmatrix.dtype).reshape(-1)  # 重排力矩阵
        
        # 如果输入的是笛卡尔坐标，需要将其转换为分数坐标
        if table_coor == 0:  
            fcoor[0:numatoms[ipoint]] = np.matmul(fcoor[0:numatoms[ipoint]], matrix)  # 乘以尺度矩阵的逆，变为分数坐标

        # 计算相对位移，四舍五入到最近的整数，以处理周期性边界条件下的跨越晶胞边界问题
        inv_coor = np.round(fcoor[0:numatoms[ipoint]] - fcoor[0])  
        
        # 进行整数位移校正，将跨越周期性边界的原子重新映射回晶胞内
        fcoor[0:numatoms[ipoint]] -= inv_coor  

        # 将分数坐标重新转换回笛卡尔坐标
        fcoor[0:numatoms[ipoint]] = np.matmul(fcoor[0:numatoms[ipoint]], scalmatrix[ipoint, :, :])  

        # 计算分子的质心
        com = np.matmul(tmpmass, fcoor[0:numatoms[ipoint], :]) / np.sum(tmpmass)  
        
        # 将当前分子的坐标转换为相对质心的坐标
        com_coor[ipoint, 0:numatoms[ipoint]] = fcoor[0:numatoms[ipoint]] - com  
    
    return com_coor, reordered_force  # 返回相对质心转换后的坐标和重排后的力矩阵（如果需要）

