import torch
import torch.nn as nn
import torchvision


class _netG(nn.Module):
    def __init__(self,opt):
        super(_netG, self).__init__()
        self.main_encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.main_decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=5,stride=2,padding=2).cuda(),
            nn.BatchNorm2d(128).cuda(),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=2).cuda(),
            nn.BatchNorm2d(64).cuda(),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,64,kernel_size=5,stride=2,padding=2).cuda(),
            nn.BatchNorm2d(64).cuda(),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,kernel_size=5,stride=2,padding=2).cuda(),
            nn.BatchNorm2d(32).cuda(),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,3,kernel_size=5,stride=2,padding=2).cuda(),
            nn.Tanh()
            
        )

    def forward(self, input):
        var = input.shape[0]
        temp = self.main_encoder(input)
        temp = torch.reshape(temp, (temp.shape[0],-1))
        temp = torch.reshape(temp, (var,9216,1)).cuda()
        filt = nn.Conv1d(9216,9216,kernel_size=1).cuda()
        temp = filt(temp)
        temp = torch.reshape(temp,(var,256,6,6))
        output = self.main_decoder(temp)
        output = torchvision.transforms.functional.resize(output,[113])
        return output


class _netlocalD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(256, 1, kernel_size=6, stride=1, ),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

