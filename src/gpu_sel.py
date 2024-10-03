import os
# 在read.py中利用nvidia-smi -q -d Memory重定向到gpu_info到获取GPU顺序的命名
# local_size变量应该是超算中自己制定的GPU使用数量
def gpu_sel(local_size):
    '''该函数的主要目的是将可用的GPU存储到系统环境变量CUDA_VISIBLE_DEVICES中'''
    # 依次提取所有GPU的可用显存(Free Mib)，存储在列表memory_gpu中
    memory_gpu=[int(x.split()[2]) for x in open('gpu_info','r').readlines()]
    if memory_gpu:
       # 生成一个按照可用显存从小到大的GPU索引列表gpu_queue
       gpu_queue=sorted(range(len(memory_gpu)), key=lambda k: memory_gpu[k],reverse=False) # reverse=False默认升序
       str_queue=""
       for i in gpu_queue[:local_size]: # 这里的i指的是对应的GPU名字，GPU0, GPU1这样
           str_queue+=str(i)
           str_queue+=", "
       os.environ['CUDA_VISIBLE_DEVICES']=str_queue[:-2] # 设置环境变量，控制具体使用的GPU, 切片移除最后的两个字符（逗号和空格）
       #string="export CUDA_VISIBLE_DEVICES='"+str_queue[:-2]+"'"
       #print(string)
       #os.system(string)
       print(os.environ.get('CUDA_VISIBLE_DEVICES'))
