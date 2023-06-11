import torch.nn as nn
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        resnet = resnet50(weights='DEFAULT')
        resnet_module = list(resnet.children())[:-5]
        self.resnet = nn.Sequential(*resnet_module)

        self.pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x = self.resnet(x)  # torch.Size([1, 256, 10, 38])
        x = self.pool(x.permute(0,3,1,2))   # torch.Size([1, 38, 256, 1])
        x = x.squeeze(-1)   # torch.Size([1, 38, 256])
        return x


class BiLSTM(nn.Module):
    def __init__(self,in_channel:int, hidden_channel:int, out_channel:int):
        super().__init__()
        self.rnn = nn.LSTM(in_channel, hidden_channel, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_channel*2, out_channel)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out= self.fc(out)
        return out


class CRNN(nn.Module):
    def __init__(self, hidden_size:int, out_channels:int):
        super().__init__()
        self.image_encoder = ResNet()
        self.sequence_modeling = nn.Sequential(
            BiLSTM(hidden_size, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, out_channels),
        )
    
    def forward(self, x):
        img_feature = self.image_encoder(x)
        sequence_feature = self.sequence_modeling(img_feature)
        return sequence_feature