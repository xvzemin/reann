import torch

class EMA():
    '''
    EMA只在预测的时候使用, 而训练的时候使用的永远是不是用EMA的参数
    EMA(Exponential Moving Average 指数移动平均)是一种用于平滑模型参数的技术
    平滑模型参数：EMA 会根据指定的衰减因子（decay）来平滑参数，这样可以减小噪声带来的影响。
    提高泛化能力：使用 EMA 生成的模型通常能在验证集或测试集上表现得更好，因为它保留了更稳定的参数值。
    减少过拟合：通过平滑参数，EMA 有助于减少模型对训练数据的过拟合
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}         # 保存 EMA 参数的字典，初始值为模型参数的克隆
        self.backup = {}         # 用于临时保存当前模型参数的字典，以便在应用或恢复 EMA 时使用
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:  # 对需要平滑的参数进行初始化
                    self.shadow[name] = param.detach().clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:  # 对需要平滑的参数进行修改
                    self.shadow[name] = (1.0 - self.decay) * param + self.decay * self.shadow[name] # self.shadow[name]是迭代的模型值 
    
    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.detach().clone()      # 备份参数
                    param.copy_(self.shadow[name])                  # 原地替换参数为平滑后的参数
    
    def restore(self):                                              # 从备份的self.backup数据恢复上一次记录的epoch参数
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])
            self.backup = {}

    def restart(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()      # 用来从已有模型继续训练时,这时的参数初始化不再是随机生成，而是从已经有的模型读取
