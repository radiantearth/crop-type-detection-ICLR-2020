import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUNet(nn.Module):
  def __init__(self, n_bands):
    super(ConvGRUNet, self).__init__()
    # 3 Conv Layers
    self.conv1 = nn.Conv2d(n_bands, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.gn1 = nn.GroupNorm(2,64)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.gn2 = nn.GroupNorm(2,64)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
    self.gn3 = nn.GroupNorm(2,128)
    # 3-layer GRU
    self.gru = nn.GRU(128, hidden_size=128, num_layers=3, batch_first=True, bidirectional = True)
    # classification layer
    self.linear = nn.Linear(128*2, 7)

  def init_hidden(self, n):
    return torch.zeros((2*3,n,128), dtype = torch.float32).cuda()
  
  def forward(self, x, mask):
    sh = x.shape
    #conv net is shared among time steps
    x = x.view(sh[0]*sh[1], sh[2], sh[3], sh[4])
    out = F.elu(self.gn1(self.conv1(x)))
    out = F.elu(self.gn2(self.conv2(out)))
    out = F.elu(self.gn3(self.conv3(out)))

    #average conv features of each time step using field mask
    out_sh = out.shape
    out = out.view(sh[0], sh[1], out_sh[1], sh[3], sh[4])
    mask = mask.view(sh[0], 1, 1, sh[3], sh[4])
    out = (out*mask).sum((3,4))/mask.sum((3,4))

    #apply GRU
    h = self.init_hidden(sh[0])
    out, h = self.gru(out, h)
    
    #use last time step for classification
    out = out[:,-1]
    out = self.linear(out)
    return out
