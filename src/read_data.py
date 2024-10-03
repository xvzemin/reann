import numpy as np

# nprob：configuration文件中提供拟合属性的维度拟合属性的维度，即有几个分量，用来切片读取configuration
# numpoint：[训练集结构数量，测试集结构数量]

def Read_data(folder_list,
              dimension_of_ab_initio_property,
              start_table=None):

    '''
    n_of_train_and_test: 一维列表
    
    ab_prop=[]           二维列表
    pbc_for_all_points:  二维列表，[训练集+数据集总点数, PBC]
    atom=[]              二维列表
    mass=[]              二维列表
    
    cell_matrix:         三维列表，每个元素内部有三个列表哦
    coor=[]              三维列表    
    force=None           三维列表

    num_atoms=[]          二维列表 每个结构具体的原子数量
 
    '''
    
    # 变量初始化部分
    coor=[]
    cell_matrix=[]  
    ab_prop=[] 
    force=None
    atom=[]
    mass=[]
    num_atoms=[]
    pbc_for_all_points=[]             
    if start_table==1:
       force=[]
    n_of_train_and_test = [0 for _ in range(len(folder_list))] # 先初始化，最终是[训练集结构数量，测试集结构数量]
    num=0                                               # 循环记数，用来记录遍历的第几个数据
    
    for idx_folder,folder in enumerate(folder_list):
        f_configuration=folder+'configuration'
        with open(f_configuration,'r') as f:
            while True:                                 # 每次读一个结构的数据，遍历configuration文件直到遇到空行
                string=f.readline() 
                if not string: break                    # configuration文件不能有空行
                                
                cell_matrix.append([])
                atom.append([])
                mass.append([])
                coor.append([])
                if start_table == 1: force.append([])                
                
                # 读取晶格常数
                string=f.readline()
                m=list(map(float,string.split()))       # 将指定的函数应用于给定的可迭代对象
                cell_matrix[num].append(m)
                string=f.readline()
                m=list(map(float,string.split()))
                cell_matrix[num].append(m)
                string=f.readline()
                m=list(map(float,string.split()))
                cell_matrix[num].append(m)
                
                # 读取PBC
                string=f.readline()
                m=list(map(float,string.split()[1:4]))
                pbc_for_all_points.append(m)
                
                # 循环读取每个结构的原子坐标，原子质量，原子质量
                while True: 
                    string=f.readline()
                    m=string.split()
                    if m[0] == "abprop:":
                        ab_prop.append(list(map(float,m[1:1+dimension_of_ab_initio_property])))
                        break
                    
                    if not start_table: # 除了start_table=0,其它执行else代码
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:])) # 将质量和坐标转化为浮点数
                        mass[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                    else:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        mass[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                        force[num].append(tmp[4:7])
                
                n_of_train_and_test[idx_folder]+=1
                num_atoms.append(len(atom[num]))
                num+=1
    return n_of_train_and_test,atom,mass,num_atoms,cell_matrix,pbc_for_all_points,coor,ab_prop,force
