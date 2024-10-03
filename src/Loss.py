import torch 
import torch.nn as nn

class Loss(nn.Module):
   def __init__(self):
      super(Loss, self).__init__()
      # 返回的是累加损失
      self.loss_fn=nn.MSELoss(reduction="sum")

   def forward(self,predictions,targets): 
      # 损失函数值是标量，view(-1)将其转化为一维张量，
      return  torch.cat([self.loss_fn(prediction,target).view(-1) for prediction, target in zip(predictions,targets)])
