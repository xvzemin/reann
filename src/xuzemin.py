class xzm:
    def __init__(self, file_name):
        self.file_name = file_name
        self.has_output = False         # 标记是否已经输出过

    def write(self, message, force_print = 0):
        if force_print == 2:           # 判断是否要强制输出
           self.has_output = False     
        if not self.has_output:
            with open(self.file_name, 'a') as f:  # 以追加模式打开文件
                f.write(message + '\n')  # 写入信息并换行
                f.flush()
            if force_print:
                self.has_output = 1  # 设置标记为已输出

    def reset(self):
        self.has_output = 0  # 允许再次输出
